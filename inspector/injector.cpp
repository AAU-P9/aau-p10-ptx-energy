#include <iostream>
#include <sstream>
#include "ASTVisitor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main(int argc, char **argv)
{
    std::stringstream buffer;
    buffer << std::cin.rdbuf();
    std::string code = buffer.str();

    ASTVisitor::runOnCode(code, "input.cu");

    /*LLVMContext Context;
    SMDiagnostic Err;

    auto M = parseIRFile("matrix_mult.ll", Err, Context);
    if (!M)
    {
        Err.print("IR Parser", errs());
        return 1;
    }

    errs() << "Successfully parsed module: " << M->getName() << "\n";*/
    return 0;
}