// parser.cpp
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
    llvm::LLVMContext context;
    llvm::SMDiagnostic err;

    std::unique_ptr<llvm::Module> module = llvm::parseIRFile("matrix_mult.ll", err, context);

    if (!module) {
        err.print("parser", errs());
        return 1;
    }

    outs() << "Module parsed successfully!\n";

    for (Function &F : *module) {
        if (F.isDeclaration()) continue;
        //outs() << "Function: " << F.getName() << "\n";
        
        //if (F.getName() != "_Z30__device_stub__gpu_matrix_multPiS_S_iii") continue;

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (CallInst *CI = dyn_cast<CallInst>(&I)) {
                    Function *Callee = CI->getCalledFunction();
                    if (!Callee) continue;

                    //printf("Hello World!\n");

                    if (Callee->getName() == "cudaLaunchKernel") {
                        outs() << "Found kernel launch in function: " << F.getName() << "\n";

                        Value* gridX    = CI->getArgOperand(1);
                        Value* blockX   = CI->getArgOperand(2);
                        Value* gridY    = CI->getArgOperand(3);
                        Value* blockY   = CI->getArgOperand(4);

                        outs() << "Grid X: "; gridX->print(outs()); outs() << "\n";
                        outs() << "Grid Y: "; gridY->print(outs()); outs() << "\n";
                        outs() << "Block X: "; blockX->print(outs()); outs() << "\n";
                        outs() << "Block Y: "; blockY->print(outs()); outs() << "\n";
                    }
                }
            }
        }
        /*if (F.getName() == "cudaLaunchKernel") {
            outs() << "Function: " << F.getName() << "\n";

            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
                        Function *Callee = CI->getCalledFunction();

                        printf("Hello World\n");
                    }
                }
            }
        }*/
    }

    return 0;
}