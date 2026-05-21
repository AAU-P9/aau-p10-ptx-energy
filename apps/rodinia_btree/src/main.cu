#include "cupti_timing.h"
#include "ptx_meta.h"

#include <cuda.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef TREE_SIZE
#define TREE_SIZE 65536
#endif

#ifndef QUERY_COUNT
#define QUERY_COUNT 1000
#endif

#ifndef ORDER
#define ORDER 256
#endif

#define DEFAULT_ORDER ORDER

// ============================================================
// Data structures
// ============================================================

typedef struct record { int value; } record;

typedef struct node {
    void **pointers;
    int *keys;
    struct node *parent;
    bool is_leaf;
    int num_keys;
    struct node *next;
} node;

typedef struct knode {
    int location;
    int indices[DEFAULT_ORDER + 1];
    int keys[DEFAULT_ORDER + 1];
    bool is_leaf;
    int num_keys;
} knode;

// ============================================================
// Globals used by tree build + transform
// ============================================================

static knode *knodes_g;
static record *krecords_g;
static char *mem_g;
static long freeptr_g;
static long malloc_size_g;
static long tree_size_g;
static node *queue_g = NULL;

static int order = DEFAULT_ORDER;

// ============================================================
// Queue helpers (BFS used by transform_to_cuda)
// ============================================================

static void enqueue(node *new_node) {
    node *c;
    if (queue_g == NULL) {
        queue_g = new_node;
        queue_g->next = NULL;
    } else {
        c = queue_g;
        while (c->next != NULL) c = c->next;
        c->next = new_node;
        new_node->next = NULL;
    }
}

static node *dequeue(void) {
    node *n = queue_g;
    queue_g = queue_g->next;
    n->next = NULL;
    return n;
}

// ============================================================
// Tree height
// ============================================================

static int tree_height(node *root) {
    int h = 0;
    node *c = root;
    while (!c->is_leaf) { c = (node *)c->pointers[0]; h++; }
    return h;
}

// ============================================================
// kmalloc: bump allocator into pre-allocated mem_g block
// ============================================================

static void *kmalloc(int sz) {
    void *r = (void *)freeptr_g;
    freeptr_g += sz;
    if (freeptr_g > malloc_size_g + (long)mem_g) {
        printf("ERROR: kmalloc overflow\n");
        exit(1);
    }
    return r;
}

// ============================================================
// transform_to_cuda: linearize pointer tree into flat arrays
// ============================================================

static long transform_to_cuda(node *root) {
    long max_nodes = (long)(pow(order, log(tree_size_g) / log(order / 2.0) - 1) + 1);
    malloc_size_g = tree_size_g * sizeof(record) + max_nodes * sizeof(knode);
    mem_g = (char *)malloc(malloc_size_g);
    if (!mem_g) { printf("ERROR: malloc failed\n"); exit(1); }
    freeptr_g = (long)mem_g;

    krecords_g = (record *)kmalloc(tree_size_g * sizeof(record));
    knodes_g   = (knode  *)kmalloc(max_nodes   * sizeof(knode));

    queue_g = NULL;
    enqueue(root);

    int i;
    long nodeindex  = 0;
    long recordindex = 0;
    long queueindex  = 0;
    knodes_g[0].location = nodeindex++;

    while (queue_g != NULL) {
        node  *n = dequeue();
        knode *k = &knodes_g[queueindex];
        k->location = queueindex++;
        k->is_leaf  = n->is_leaf;
        k->num_keys = n->num_keys + 2;

        k->keys[0] = INT_MIN;
        k->keys[k->num_keys - 1] = INT_MAX;
        for (i = k->num_keys; i < order; i++) k->keys[i] = INT_MAX;

        if (!k->is_leaf) {
            k->indices[0] = nodeindex++;
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                enqueue((node *)n->pointers[i - 1]);
                k->indices[i] = nodeindex++;
            }
            enqueue((node *)n->pointers[i - 1]);
        } else {
            k->indices[0] = 0;
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                krecords_g[recordindex].value = ((record *)n->pointers[i - 1])->value;
                k->indices[i] = recordindex++;
            }
        }
        k->indices[k->num_keys - 1] = queueindex;
    }

    return tree_size_g * sizeof(record) + nodeindex * sizeof(knode);
}

// ============================================================
// B+ tree insertion helpers
// ============================================================

static record *make_record(int value) {
    record *r = (record *)malloc(sizeof(record));
    r->value = value;
    return r;
}

static node *make_node(void) {
    node *n = (node *)malloc(sizeof(node));
    n->keys     = (int   *)malloc((order - 1) * sizeof(int));
    n->pointers = (void **)malloc( order       * sizeof(void *));
    n->is_leaf  = false;
    n->num_keys = 0;
    n->parent   = NULL;
    n->next     = NULL;
    return n;
}

static node *make_leaf(void) {
    node *leaf = make_node();
    leaf->is_leaf = true;
    return leaf;
}

static node *find_leaf(node *root, int key, bool verbose) {
    (void)verbose;
    int i = 0;
    node *c = root;
    if (!c) return c;
    while (!c->is_leaf) {
        i = 0;
        while (i < c->num_keys) { if (key >= c->keys[i]) i++; else break; }
        c = (node *)c->pointers[i];
    }
    return c;
}

static record *find(node *root, int key, bool verbose) {
    int i = 0;
    node *c = find_leaf(root, key, verbose);
    if (!c) return NULL;
    for (i = 0; i < c->num_keys; i++) if (c->keys[i] == key) break;
    return (i == c->num_keys) ? NULL : (record *)c->pointers[i];
}

static int cut(int length) { return (length % 2 == 0) ? length / 2 : length / 2 + 1; }

static int get_left_index(node *parent, node *left) {
    int li = 0;
    while (li <= parent->num_keys && parent->pointers[li] != left) li++;
    return li;
}

static node *insert_into_leaf(node *leaf, int key, record *pointer) {
    int i, ins = 0;
    while (ins < leaf->num_keys && leaf->keys[ins] < key) ins++;
    for (i = leaf->num_keys; i > ins; i--) {
        leaf->keys[i]     = leaf->keys[i - 1];
        leaf->pointers[i] = leaf->pointers[i - 1];
    }
    leaf->keys[ins]     = key;
    leaf->pointers[ins] = pointer;
    leaf->num_keys++;
    return leaf;
}

/* forward declarations */
static node *insert_into_parent(node *root, node *left, int key, node *right);

static node *insert_into_leaf_after_splitting(node *root, node *leaf, int key, record *pointer) {
    node *new_leaf = make_leaf();
    int *temp_keys     = (int   *)malloc(order * sizeof(int));
    void **temp_ptrs   = (void **)malloc(order * sizeof(void *));
    int ins = 0;
    while (leaf->keys[ins] < key && ins < order - 1) ins++;
    int i, j;
    for (i = 0, j = 0; i < leaf->num_keys; i++, j++) {
        if (j == ins) j++;
        temp_keys[j] = leaf->keys[i];
        temp_ptrs[j] = leaf->pointers[i];
    }
    temp_keys[ins] = key;
    temp_ptrs[ins] = pointer;
    leaf->num_keys = 0;
    int split = cut(order - 1);
    for (i = 0; i < split; i++) { leaf->pointers[i] = temp_ptrs[i]; leaf->keys[i] = temp_keys[i]; leaf->num_keys++; }
    for (i = split, j = 0; i < order; i++, j++) { new_leaf->pointers[j] = temp_ptrs[i]; new_leaf->keys[j] = temp_keys[i]; new_leaf->num_keys++; }
    free(temp_ptrs); free(temp_keys);
    new_leaf->pointers[order - 1] = leaf->pointers[order - 1];
    leaf->pointers[order - 1] = new_leaf;
    for (i = leaf->num_keys;     i < order - 1; i++) leaf->pointers[i]     = NULL;
    for (i = new_leaf->num_keys; i < order - 1; i++) new_leaf->pointers[i] = NULL;
    new_leaf->parent = leaf->parent;
    return insert_into_parent(root, leaf, new_leaf->keys[0], new_leaf);
}

static node *insert_into_node(node *root, node *n, int left_index, int key, node *right) {
    int i;
    for (i = n->num_keys; i > left_index; i--) { n->pointers[i + 1] = n->pointers[i]; n->keys[i] = n->keys[i - 1]; }
    n->pointers[left_index + 1] = right;
    n->keys[left_index] = key;
    n->num_keys++;
    return root;
}

static node *insert_into_node_after_splitting(node *root, node *old_node, int left_index, int key, node *right) {
    int i, j, split, k_prime;
    node **temp_ptrs = (node **)malloc((order + 1) * sizeof(node *));
    int  *temp_keys  = (int   *)malloc( order       * sizeof(int));
    for (i = 0, j = 0; i < old_node->num_keys + 1; i++, j++) { if (j == left_index + 1) j++; temp_ptrs[j] = (node *)old_node->pointers[i]; }
    for (i = 0, j = 0; i < old_node->num_keys;     i++, j++) { if (j == left_index)     j++; temp_keys[j] = old_node->keys[i]; }
    temp_ptrs[left_index + 1] = right;
    temp_keys[left_index]     = key;
    split = cut(order);
    node *new_node = make_node();
    old_node->num_keys = 0;
    for (i = 0; i < split - 1; i++) { old_node->pointers[i] = temp_ptrs[i]; old_node->keys[i] = temp_keys[i]; old_node->num_keys++; }
    old_node->pointers[i] = temp_ptrs[i];
    k_prime = temp_keys[split - 1];
    for (++i, j = 0; i < order; i++, j++) { new_node->pointers[j] = temp_ptrs[i]; new_node->keys[j] = temp_keys[i]; new_node->num_keys++; }
    new_node->pointers[j] = temp_ptrs[i];
    free(temp_ptrs); free(temp_keys);
    new_node->parent = old_node->parent;
    node *child;
    for (i = 0; i <= new_node->num_keys; i++) { child = (node *)new_node->pointers[i]; child->parent = new_node; }
    return insert_into_parent(root, old_node, k_prime, new_node);
}

static node *insert_into_new_root(node *left, int key, node *right) {
    node *root = make_node();
    root->keys[0]     = key;
    root->pointers[0] = left;
    root->pointers[1] = right;
    root->num_keys++;
    root->parent = NULL;
    left->parent  = root;
    right->parent = root;
    return root;
}

static node *insert_into_parent(node *root, node *left, int key, node *right) {
    node *parent = left->parent;
    if (!parent) return insert_into_new_root(left, key, right);
    int left_index = get_left_index(parent, left);
    if (parent->num_keys < order - 1) return insert_into_node(root, parent, left_index, key, right);
    return insert_into_node_after_splitting(root, parent, left_index, key, right);
}

static node *start_new_tree(int key, record *pointer) {
    node *root = make_leaf();
    root->keys[0]            = key;
    root->pointers[0]        = pointer;
    root->pointers[order - 1] = NULL;
    root->parent  = NULL;
    root->num_keys = 1;
    return root;
}

static node *insert(node *root, int key, int value) {
    if (find(root, key, false)) return root;
    record *pointer = make_record(value);
    if (!root) return start_new_tree(key, pointer);
    node *leaf = find_leaf(root, key, false);
    if (leaf->num_keys < order - 1) { insert_into_leaf(leaf, key, pointer); return root; }
    return insert_into_leaf_after_splitting(root, leaf, key, pointer);
}

// ============================================================
// GPU kernel: findK (one block per query, ORDER threads/block)
// ============================================================

__global__ void findK(long height,
                      knode *knodesD, long knodes_elem,
                      record *recordsD,
                      long *currKnodeD, long *offsetD,
                      int *keysD, record *ansD)
{
    int thid = threadIdx.x;
    int bid  = blockIdx.x;

    META_LOOP(descent_loop, 1, 20, false);
    for (int i = 0; i < height; i++) {
        if (knodesD[currKnodeD[bid]].keys[thid]     <= keysD[bid] &&
            knodesD[currKnodeD[bid]].keys[thid + 1]  > keysD[bid]) {
            if (knodesD[offsetD[bid]].indices[thid] < knodes_elem)
                offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
        }
        __syncthreads();
        if (thid == 0) currKnodeD[bid] = offsetD[bid];
        __syncthreads();
    }
    if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid])
        ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
}

// ============================================================
// main
// ============================================================

int main(void) {
    METRICS_KERNEL_START

    // --------------------------------------------------------
    // Build B+ tree on CPU
    // --------------------------------------------------------
    tree_size_g = TREE_SIZE;
    node *root = NULL;
    for (int i = 0; i < TREE_SIZE; i++)
        root = insert(root, i, i);

    long mem_used = transform_to_cuda(root);
    long maxheight = tree_height(root);
    long rootLoc = (long)knodes_g - (long)mem_g;

    int count = QUERY_COUNT;

    record *records_cpu  = (record *)mem_g;
    long records_mem     = rootLoc;

    knode *knodes_cpu    = (knode *)((long)mem_g + rootLoc);
    long knodes_elem     = ((long)mem_used - rootLoc) / (long)sizeof(knode);
    long knodes_mem      = (long)mem_used - rootLoc;

    // --------------------------------------------------------
    // Generate random query keys in [0, TREE_SIZE)
    // --------------------------------------------------------
    int *keys = (int *)malloc(count * sizeof(int));
    srand(42);
    for (int i = 0; i < count; i++)
        keys[i] = (int)((rand() / (float)RAND_MAX) * TREE_SIZE);

    long *currKnode = (long *)calloc(count, sizeof(long));
    long *offset    = (long *)calloc(count, sizeof(long));
    record *ans     = (record *)malloc(count * sizeof(record));
    for (int i = 0; i < count; i++) ans[i].value = -1;

    // --------------------------------------------------------
    // GPU allocations and H2D copies
    // --------------------------------------------------------
    record *recordsD; cudaMalloc(&recordsD, records_mem);
    cudaMemcpy(recordsD, records_cpu, records_mem, cudaMemcpyHostToDevice);

    knode *knodesD; cudaMalloc(&knodesD, knodes_mem);
    cudaMemcpy(knodesD, knodes_cpu, knodes_mem, cudaMemcpyHostToDevice);

    long *currKnodeD; cudaMalloc(&currKnodeD, count * sizeof(long));
    cudaMemcpy(currKnodeD, currKnode, count * sizeof(long), cudaMemcpyHostToDevice);

    long *offsetD; cudaMalloc(&offsetD, count * sizeof(long));
    cudaMemcpy(offsetD, offset, count * sizeof(long), cudaMemcpyHostToDevice);

    int *keysD; cudaMalloc(&keysD, count * sizeof(int));
    cudaMemcpy(keysD, keys, count * sizeof(int), cudaMemcpyHostToDevice);

    record *ansD; cudaMalloc(&ansD, count * sizeof(record));
    cudaMemcpy(ansD, ans, count * sizeof(record), cudaMemcpyHostToDevice);

    // --------------------------------------------------------
    // Launch findK
    // --------------------------------------------------------
    dim3 grid(count);
    dim3 block(ORDER);

    findK<<<grid, block>>>(maxheight, knodesD, knodes_elem, recordsD,
                           currKnodeD, offsetD, keysD, ansD);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  count);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", ORDER);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    cudaFree(recordsD); cudaFree(knodesD);
    cudaFree(currKnodeD); cudaFree(offsetD);
    cudaFree(keysD); cudaFree(ansD);
    free(keys); free(currKnode); free(offset); free(ans);
    free(mem_g);

    return 0;
}
