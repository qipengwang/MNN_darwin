//
//  Executor.hpp
//  MNN
//
//  Created by MNN on 2019/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef Executor_hpp
#define Executor_hpp
#include <MNN/ErrorCode.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/Interpreter.hpp>
#include <vector>
#include <mutex>
#include <set>
#include <MNN/MNNForwardType.h>
namespace MNN {
//下面的这些东西可以当作ptr使用，而非obj
//或者就是告诉下面定义的executor有这些类
class Backend;
class Execution;
class Runtime; // backend.hpp
struct Op;
namespace Express {
class MNN_PUBLIC Executor {
public:
    class ComputeCache;
    struct Unit;
    /*
     * 每个expr都对应一个Unit，Unit里面包含了Op指针，也就是对应的Op操作
     * 每个layer会产生很多个expr，这些expr对应很多个Unit，这些Unit会被同一个cache进行打包
     * expr->inside->mCache 是通过shared_ptr进行共享的
     * */
    static void setShapeDirty(ComputeCache* cache);
    static void setContentDirty(ComputeCache* cache);
    static void* mapOutput(ComputeCache* cache, int offset, Tensor* dest);
    struct Requirement {
        std::vector<bool> contentNeedContent;
        std::vector<bool> shapeNeedContent;
    };
    ~Executor();
    Requirement getRequirement(Expr* expr) const;
    ErrorCode computeInfo(Expr* expr);
    void makeCache(const std::vector<EXPRP>& expr, bool forceCPU = false);
    ErrorCode runCache(std::shared_ptr<ComputeCache> cache);
    void setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread);
    enum GCFlag {
        FULL,
        PART
    };
    void gc(GCFlag flag = FULL);
    static std::shared_ptr<Executor> getGlobalExecutor();

    static std::shared_ptr<Executor> newExecutor(MNNForwardType type,
                                                 const BackendConfig& config,
                                                 int numberThread);
    void resetProfile();
    void dumpProfile();
    void addOpCostTime(int op, float costTime);
    void addOpCostTime(const std::string& type, float costTime);
    void addOpFlops(const std::string& type, float flops);
    class Profiler;
    static RuntimeInfo getRuntime();
private:
    void _makeCache(const std::vector<EXPRP>& outputs, bool forceCPU);
    void _create(const std::vector<EXPRP>& outputs, std::set<std::shared_ptr<Executor::ComputeCache>>&& inputCaches, std::set<std::shared_ptr<Expr::Inside>>&& inputNode, bool forceCPU);

    void _visit(EXPRP expr, std::set<std::shared_ptr<Executor::ComputeCache>>& inputCaches, std::set<std::shared_ptr<Expr::Inside>>& inputNode);

    Executor(std::shared_ptr<Runtime> backend, MNNForwardType type);
    std::pair<std::shared_ptr<Runtime>, MNNForwardType> mRuntime;
    std::pair<std::shared_ptr<Runtime>, MNNForwardType> mBackupRuntime;
    std::mutex mMutex;
    std::shared_ptr<Profiler> mProfiler;
};
} // namespace Express
} // namespace MNN
#endif
