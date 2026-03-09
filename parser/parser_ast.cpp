#include <fstream>
#include <sstream>
#include <string>
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendAction.h"

using namespace clang;

class MyVisitor : public RecursiveASTVisitor<MyVisitor> {
public:
    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (FD->hasAttr<CUDAGlobalAttr>()) {
            llvm::outs() << "Found __global__ CUDA kernel: " << FD->getNameAsString() << "\n";

            for (unsigned i = 0; i < FD->getNumParams(); ++i) {
                ParmVarDecl *Param = FD->getParamDecl(i);
                llvm::outs() << "Parameter: " << Param->getNameAsString()
                             << " of type: " 
                             << Param->getType().getAsString() << "\n";
            }
        }
        return true;
    }
};

class MyASTConsumer : public ASTConsumer {
public:
    bool HandleTopLevelDecl(DeclGroupRef DG) override {
        for (auto D : DG)
            Visitor.TraverseDecl(D);
        return true;
    }
private:
    MyVisitor Visitor;
};

class MyFrontendAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef file) override {
        return std::make_unique<MyASTConsumer>();
    }
};

int main(int argc, const char **argv) {
    if (argc < 2) {
        llvm::errs() << "Usage: " << argv[0] << " <source-file>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        llvm::errs() << "Could not open file: " << argv[1] << "\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string code = buffer.str();

    clang::tooling::runToolOnCode(
        std::make_unique<MyFrontendAction>(),
        code,
        argv[1]
    );

    return 0;
}