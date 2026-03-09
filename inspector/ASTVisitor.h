#ifndef ASTVISITOR_H
#define ASTVISITOR_H

#include <string>
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"

class ASTVisitor
{
public:
    class Visitor : public clang::RecursiveASTVisitor<Visitor>
    {
    public:
        Visitor(clang::Rewriter &R) : TheRewriter(R) {}

        std::vector<std::pair<std::string,std::string>> KernelParams;

        bool VisitFunctionDecl(clang::FunctionDecl *FD)
        {
            if (FD->hasAttr<clang::CUDAGlobalAttr>() && FD->hasBody())
            {
                for (unsigned i = 0; i < FD->getNumParams(); ++i)
                {
                    clang::ParmVarDecl *Param = FD->getParamDecl(i);
                    KernelParams.emplace_back(Param->getType().getAsString(), Param->getNameAsString());
                }

                /*llvm::outs() << "Found __global__ CUDA kernel: " << FD->getNameAsString() << "\n";

                for (unsigned i = 0; i < FD->getNumParams(); ++i)
                {
                    clang::ParmVarDecl *Param = FD->getParamDecl(i);
                    llvm::outs() << "Parameter: " << Param->getNameAsString()
                                 << " of type: "
                                 << Param->getType().getAsString() << "\n";
                }*/
            }
            return true;
        }

    private:
        clang::Rewriter &TheRewriter;
    };

    class Consumer : public clang::ASTConsumer
    {
    public:
        Consumer(clang::Rewriter &R) : VisitorInstance(R), TheRewriter(R) {}

        void HandleTranslationUnit(clang::ASTContext &Context) override
        {
            clang::SourceManager &SM = TheRewriter.getSourceMgr();
            clang::FileID FID = SM.getMainFileID();
            llvm::StringRef CodeRef = SM.getBufferData(FID);

            // Adding #include "lli.h" to build folder
            std::string Code = CodeRef.str();
            size_t Pos = Code.find("#include <cuda.h>");
            if (Pos != std::string::npos)
            {
                size_t LineEnd = Code.find("\n", Pos);
                if (LineEnd != std::string::npos)
                {
                    clang::SourceLocation InsertLoc = SM.getLocForStartOfFile(FID).getLocWithOffset(LineEnd + 1);
                    TheRewriter.InsertText(InsertLoc, "#include \"../lli.h\"\n", true, true);
                }
            }

            VisitorInstance.TraverseDecl(Context.getTranslationUnitDecl());

            clang::SourceLocation EndLoc = SM.getLocForEndOfFile(FID);
            std::string MadsenFunctionCode = GenerateMadsenFunctionCode(VisitorInstance.KernelParams);
            TheRewriter.InsertText(EndLoc, MadsenFunctionCode, true, true);

            VisitorInstance.TraverseDecl(Context.getTranslationUnitDecl());
        }

    private:
        Visitor VisitorInstance;
        clang::Rewriter &TheRewriter;
    };

    class Action : public clang::ASTFrontendAction
    {
    public:
        Action() = default;

        std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                              clang::StringRef file) override
        {
            TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
            return std::make_unique<Consumer>(TheRewriter);
        }

        void EndSourceFileAction() override
        {
            TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
                .write(llvm::outs());
        }

    private:
        clang::Rewriter TheRewriter;
    };

    static std::string GenerateMadsenFunctionCode(const std::vector<std::pair<std::string, std::string>> &params)
    {
        std::string code = "void madsen_function(void **args) {\n";
        for (size_t i = 0; i < params.size(); ++i)
        {
            const auto &type_name = params[i].first;
            const auto &param_name = params[i].second;

            code += "    printf(\"Parameter " + param_name + " = ";
            if (type_name == "int")
                code += "%d";
            else if (type_name == "float")
                code += "%f";
            else if (type_name == "double")
                code += "%lf";
            else
                code += "%p";
            code += "\\n\", *(" + type_name + "*)args[" + std::to_string(i) + "]);\n";
        }
        code += "    exit(0);\n";

        code += "}\n";
        return code;
    }

    static void runOnCode(const std::string &Code, const std::string &FileName = "input.cu")
    {
        clang::tooling::runToolOnCode(std::make_unique<Action>(), Code, FileName);
    }
};

#endif // ASTVISITOR_H