#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
    LLVMContext Context;
    SMDiagnostic Err;

    std::unique_ptr<Module> M = parseIRFile("main.ll", Err, Context);

    if (!M) {
        Err.print("IR Parser", errs());
        return 1;
    }

    for (Function &F : *M) {
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                llvm::outs() << I << "\n";
            }
        }
    }
}