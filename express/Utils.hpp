//
//  Utils.hpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Utils_hpp
#define Utils_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/Tensor.hpp>
#include "Type_generated.h"
#include <MNN/expr/Executor.hpp>
namespace MNN {
namespace Express {
struct Expr::Inside {
    /*
     * 每个expr都对应一个Unit，Unit里面包含了Op指针，也就是对应的Op操作
     * 每个layer会产生很多个expr，这些expr对应很多个Unit，这些Unit会被同一个cache进行打包
     * expr->inside->mCache 是通过shared_ptr进行共享的
     * */
    Inside(int outputSize);
    ~ Inside();
    std::vector<Variable::Info> mOutputInfos;
    std::vector<Tensor*> mOutputTensors;
    Executor::Requirement mReq;
    std::shared_ptr<Executor::Unit> mUnit;
    std::shared_ptr<Executor::ComputeCache> mCache;
    int mCacheOffset = 0;
    bool mInfoDirty = true; // 对应的info是不全的或者需要修改的，即当前的info是不对的
    bool mContentDirty = true; // 对应的content被修改过了
};
class MNN_PUBLIC Utils {
public:
    static void copyInfoToTensor(Tensor* dest, const Variable::Info* source);
    static void copyTensorToInfo(Variable::Info* dest, const Tensor* source);
    static DataType convertDataType(halide_type_t type);
    static int convertFormat(Dimensionformat format);
    static Express::Dimensionformat revertFormat(int format);
    static halide_type_t revertDataType(DataType dataType);
    static bool allocMemoryForHostTensor(Tensor* dest);
    static bool releaseMemoryForHostTensor(Tensor* dest);
};
} // namespace Express
} // namespace MNN
#endif
