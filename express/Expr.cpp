//
//  Expr.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define FLATBUFFERS_PREFER_PRINTF

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <map>
#include "Utils.hpp"
#include "core/FileLoader.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
//#define MNN_OPEN_TIME_TRACE
#include "MNN/AutoTime.hpp"
#include "MNN/expr/ExecutorScope.hpp"

//#define MNN_EXPRESS_ERROR_REPORT
static inline std::string numberToString(int index) {
    char s[10];
    snprintf(s, 10, "%d", index);
    return std::string(s);
}

static bool HasUnknownDim(const std::vector<int>& dims) {
    for (const int& dim : dims) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

namespace MNN {
namespace Express {
void Variable::Info::syncSize() {
    size = 1;
    for (int i=0; i<dim.size(); ++i) {
        if (dim[i] <= 0) {
            // Not valid
            size = 0;
            return;
        }
        if (order == NC4HW4 && i == 1) {
            size *= (UP_DIV(dim[1], 4) * 4);
        } else {
            size *= dim[i];
        }
    }
}

bool VARP::fix(VARP::InputType type) const {
    if (nullptr == mContent->expr().first->get()) {
        mContent->expr().first->mType = type;
        return true;
    }
    auto info = mContent->getInfo();
    if (nullptr == info) {
        return false;
    }
    VARP newVar;
    switch (type) {
        case INPUT: {
            newVar = _Input(info->dim, info->order, info->type);
            auto ptr = mContent->readMap<void>();
            if (nullptr != ptr) {
                auto dstPtr = newVar->writeMap<void>();
                ::memcpy(dstPtr, ptr, info->size * info->type.bytes());
            }
            break;
        }
        case CONSTANT: {
            auto ptr = mContent->readMap<void>();
            if (nullptr == ptr) {
                return false;
            }
            newVar = _Const(ptr, info->dim, info->order, info->type);
            break;
        }
        case TRAINABLE: {
            auto ptr = mContent->readMap<void>();
            if (nullptr == ptr) {
                return false;
            }
            newVar = _TrainableParam(ptr, info->dim, info->order, info->type);
            break;
        }
        default:
            return false;
    }
    Variable::replace(VARP(mContent), newVar);
    return true;
}

Expr::Expr(int outputSize) {
    mInside.reset(new Inside(outputSize));
    mOutputNames.resize(outputSize);
}

Expr::~Expr() {
    mInside.reset();
}
Variable::Info* Expr::outputInfo(int index) const {
    return mInside->mOutputInfos.data() + index;
}

void Expr::_addLinkForInputs(EXPRP expr) {
    auto inputs = expr->inputs();
    for (int i=0; i<inputs.size(); ++i) {
        bool findEmpty = false;
        auto inputExpr = inputs[i]->mFrom;
        for (int j=0; j<inputExpr->mTo.size(); ++j) {
            auto ref = inputExpr->mTo[j].lock();
            if (nullptr == ref) {
                inputExpr->mTo[j] = WeakEXPRP(expr);
                findEmpty = true;
                break;
            }
        }
        if (!findEmpty) {
            inputExpr->mTo.emplace_back(WeakEXPRP(expr));
        }
    }
}
EXPRP Expr::create(Variable::Info&& info, const void* ptr, VARP::InputType type, bool copy, std::string name) {
    EXPRP expr(new Expr(1));
    expr->mName = name;
    expr->mOp = nullptr;
    auto originPtr = ptr;
    expr->mInside->mOutputInfos[0] = std::move(info);
    auto& dstInfo = expr->mInside->mOutputInfos[0];
    expr->mInside->mInfoDirty = false;
    dstInfo.syncSize();
    Utils::copyInfoToTensor(expr->mInside->mOutputTensors[0], expr->mInside->mOutputInfos.data());
    expr->mType = type;
    if (type == VARP::CONSTANT) {
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::CONSTANT;
    } else if (type == VARP::INPUT) {
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::INPUT;
    } else {
        // VARP::TRAINABLE
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->usage = Tensor::InsideDescribe::TRAINABLE;
    }
    if (dstInfo.size > 0 && copy) {
        auto res = Utils::allocMemoryForHostTensor(expr->mInside->mOutputTensors[0]);
        if (!res) {
            MNN_ASSERT(false);
            return nullptr;
        }
    } else {
        expr->mInside->mOutputTensors[0]->buffer().host = nullptr;
    }
    if (nullptr == originPtr) {
        if (type == VARP::INPUT && dstInfo.size > 0) {
            expr->mInside->mContentDirty = true;
        }
        return expr;
    }
    expr->mInside->mContentDirty = false;
    if (copy) {
        ::memcpy(expr->mInside->mOutputTensors[0]->buffer().host, originPtr, dstInfo.size * dstInfo.type.bytes());
    } else {
        TensorUtils::getDescribe(expr->mInside->mOutputTensors[0])->memoryType = Tensor::InsideDescribe::MEMORY_OUTSIDE;
        expr->mInside->mOutputTensors[0]->buffer().host = (uint8_t*)originPtr;
    }
    return expr;
}
EXPRP Expr::create(std::pair<std::shared_ptr<char>, int> extra, std::vector<VARP>&& inputs, int outputSize, std::string name) {
    EXPRP expr(new Expr(outputSize));
    expr->setName(name);
    expr->mExtraBuffer = extra.first;
    expr->mOpBufferSize = extra.second;
    expr->mOp = flatbuffers::GetMutableRoot<Op>(extra.first.get());
    expr->mOpBufferSize = extra.second;
    expr->mInputs   = std::move(inputs);
    expr->mInside->mReq = ExecutorScope::Current()->getRequirement(expr.get());
    _addLinkForInputs(expr);
    return expr;
}

EXPRP Expr::create(const OpT* op, std::vector<VARP> inputs, int outputSize, std::string name) {
    if (OpType_Input == op->type) {
        Variable::Info info;
        info.dim = op->main.AsInput()->dims;
        if (info.dim.size() >= 1 && -1 == info.dim[0]) {
            info.dim[0] = 1;
        }
        info.order = Utils::revertFormat(op->main.AsInput()->dformat);
        info.type = Utils::revertDataType(op->main.AsInput()->dtype);
        return create(std::move(info), nullptr, VARP::INPUT, true, name);
    }
    if (OpType_Const == op->type || OpType_TrainableParam == op->type) {
        Variable::Info info;
        info.dim = op->main.AsBlob()->dims;
        info.order = Utils::revertFormat(op->main.AsBlob()->dataFormat);
        void* ptr = nullptr;
        info.type = Utils::revertDataType(op->main.AsBlob()->dataType);
        switch (op->main.AsBlob()->dataType) {
            case DataType_DT_INT8:
                ptr = (void*)op->main.AsBlob()->int8s.data();
                break;
            case DataType_DT_INT32:
                ptr = (void*)op->main.AsBlob()->int32s.data();
                break;
            case DataType_DT_UINT8:
                ptr = (void*)op->main.AsBlob()->uint8s.data();
                break;
            case DataType_DT_FLOAT:
                ptr = (void*)op->main.AsBlob()->float32s.data();
                break;
            default:
                break;
        }
        //MNN_ASSERT(nullptr != ptr);
        auto expr = create(std::move(info), ptr, VARP::CONSTANT, true, name);
        if (OpType_TrainableParam == op->type && nullptr != ptr) {
            expr->mType = VARP::TRAINABLE;
        }
        return expr;
    }
    flatbuffers::FlatBufferBuilder builder;
    auto offset = Op::Pack(builder, op);
    builder.Finish(offset);
    std::shared_ptr<char> extraBuffer(new char[builder.GetSize()], std::default_delete<char[]>());
    ::memcpy(extraBuffer.get(), builder.GetBufferPointer(), builder.GetSize());
    auto resExpr = Expr::create(std::make_pair(extraBuffer, builder.GetSize()), std::move(inputs), outputSize);
    if(op->name.empty()) {
        resExpr->setName(name);
    }else {
        resExpr->setName(op->name);
    }
    return resExpr;
}
void Expr::setName(const std::string& name) {
    mName = name;
}
bool Expr::requireInfo() {
    if (!mInside->mInfoDirty) {
        return true;
    }
    if (!mValid) {
        return false;
    }
    if (nullptr == mOp) {
        return !HasUnknownDim(mInside->mOutputInfos[0].dim);
    }
    bool ready     = true;
    for (int i = 0; i < mInputs.size(); ++i) {
        if (nullptr == mInputs[i] || nullptr == mInputs[i]->mFrom) {
            // The Variable is set nullptr by api
            return false;
        }
        auto inputInfo = mInputs[i]->getInfo();
        if (nullptr == inputInfo) {
#ifdef MNN_EXPRESS_ERROR_REPORT
            MNN_ERROR("%s, %d input not ready\n", mName.c_str(), i);
#endif
            mValid = false;
            return false;
        }
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        auto& v  = mInputs[i];
        if (mInside->mReq.shapeNeedContent[i]) {
            // For shape need content, the content must not be nullptr
            auto ptr = v->readInternal(true);
            if (nullptr == ptr) {
                ready = false;
                break;
            }
        }
    }
    if (!ready) {
        return false;
    }
    //MNN_PRINT("Info %s, %p Start\n", mName.c_str(), this);
    auto res   = ExecutorScope::Current()->computeInfo(this);
    //MNN_PRINT("Info Compute %s\n", mName.c_str());

    if (NO_ERROR == res) {
        mInside->mInfoDirty = false;
    } else {
        mValid = false;
    }
    return NO_ERROR == res;
}

size_t Variable::linkNumber() const {
    return mFrom->outputs().size();
}
const std::vector<WeakEXPRP>& Variable::toExprs() const {
    return mFrom->outputs();
}

VARP Variable::create(EXPRP expr, int index) {
    VARP res(new Variable(expr, index));
#ifdef MNN_EXPR_SHAPE_EAGER
    auto info = expr->requireInfo();
    if (!info) {
#ifdef MNN_EXPRESS_ERROR_REPORT
        MNN_ERROR("Can't compute shape\n");
#endif
    }
#endif
    return res;
}
void Expr::replace(EXPRP old, EXPRP from) {
    if (old.get() == from.get()) {
        return;
    }
    for (auto input : old->inputs()) {
        for (int j=0; j<input->mFrom->mTo.size(); ++j) {
            auto ref = input->mFrom->mTo[j].lock();
            if (ref.get() == old.get()) {
                input->mFrom->mTo[j].reset();
            }
        }
    }
    for (auto input : from->inputs()) {
        bool hasSet = false;
        for (int j=0; j<input->mFrom->mTo.size(); ++j) {
            auto ref = input->mFrom->mTo[j].lock();
            if (ref.get() == old.get()) {
                hasSet = true;
                break;
            }
        }
        if (!hasSet) {
            for (int j=0; j<input->mFrom->mTo.size(); ++j) {
                auto ref = input->mFrom->mTo[j].lock();
                if (nullptr == ref) {
                    input->mFrom->mTo[j] = WeakEXPRP(old);
                    hasSet = true;
                    break;
                }
            }
        }
        if (!hasSet) {
            input->mFrom->mTo.emplace_back(WeakEXPRP(old));
        }
    }
    old->mOp = from->mOp;
    old->mName = from->mName;
    old->mOutputNames = from->mOutputNames;
    old->mExtraBuffer = from->mExtraBuffer;
    old->mOpBufferSize = from->mOpBufferSize;
    old->mType = from->mType;
    old->mValid = from->mValid;
    old->mInside = from->mInside;
    old->mInputs = from->mInputs;
    std::vector<Expr*> visited;
    old->visitOutputs([&](EXPRP expr, int index) {
        if (expr->visited()) {
            return false;
        }
        visited.emplace_back(expr.get());
        expr->setVisited(true);
        expr->mInside->mCache.reset();
        expr->mInside->mCacheOffset = 0;
        expr->mValid = true;
        expr->mInside->mInfoDirty = true;
        return true;
    });
    for (auto e : visited) {
        e->setVisited(false);
    }
}

void Variable::setName(const std::string& name) {
    mFrom->mOutputNames[mFromIndex] = name;
    if (mFrom->name().empty()) {
        mFrom->setName(name);
    }
}
const std::string& Variable::name() const {
    return mFrom->outputName(mFromIndex);
}
bool Variable::input(VARP src) {
    if (nullptr != mFrom->get() || VARP::CONSTANT == mFrom->mType) {
        MNN_ERROR("Can't input to no-input op\n");
        return false;
    }
    if (nullptr == src) {
        /*Close the Input*/
        mFrom->visitOutputs([](EXPRP expr, int index) {
            auto recurse = expr->mValid; expr->mValid = false;
            return recurse;
        });
        mFrom->mValid = false;
        return false;
    }
    auto info = src->getInfo();
    std::shared_ptr<Variable::Info> tempInfo;
    if (nullptr == info) {
        tempInfo.reset(new Variable::Info);
        tempInfo->size = 0;
        tempInfo->type = halide_type_of<float>();
        info = tempInfo.get();
    }
    auto dstInfo = getInfo();
    bool needChange = nullptr == dstInfo || info->order != dstInfo->order || info->dim.size() != dstInfo->dim.size() || info->type != dstInfo->type;
    if (!needChange) {
        for (int i=0; i<info->dim.size(); ++i) {
            if (dstInfo->dim[i] != info->dim[i]) {
                needChange = true;
                break;
            }
        }
    }

    if (!mFrom->mInside->mCache) {
        ExecutorScope::Current()->makeCache({mFrom}, false);
    }
    if (needChange) {
        mFrom->mInside->mOutputInfos[0] = *info;
        Utils::releaseMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
        Utils::copyInfoToTensor(mFrom->inside()->mOutputTensors[0], mFrom->inside()->mOutputInfos.data());
        Utils::allocMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    }
    if (info->size) {
        auto dstPtr = writeInternal(false);
        auto srcPtr = src->readMap<void>();
        if (nullptr == dstPtr || nullptr == srcPtr) {
            //MNN_ERROR("Alloc memory error or compute src error in Variable::Input\n");
            return false;
        }
        ::memcpy(dstPtr, srcPtr, info->size * info->type.bytes());
    }
    if (needChange) {
        mFrom->visitOutputs([](EXPRP expr, int index) { return expr->setInfoDirty(); });
    } else {
        informDirty();
    }
    mFrom->mInside->mContentDirty = false;
    return true;
}

void Variable::replace(VARP dst, VARP src) {
    if (nullptr == src) {
        dst->setExpr(nullptr, 0);
        return;
    }
    if (nullptr == dst) {
        dst.mContent = src.mContent;
        return;
    }
    if (src->mFrom.get() == dst->mFrom.get()) {
        dst->mFromIndex = src->mFromIndex;
        return;
    }
    if (src->mFrom->outputSize() != dst->mFrom->outputSize()) {
        // Can't replace Expr, Just replace VARP
        std::vector<Expr*> visited;
        dst->mFrom->visitOutputs([src, dst, &visited](EXPRP expr, int index) {
            if (expr->visited()) {
                return false;
            }
            expr->setVisited(true);
            visited.emplace_back(expr.get());
            expr->mInside->mCache.reset();
            expr->mInside->mCacheOffset = 0;
            expr->mValid = true;
            expr->mInside->mInfoDirty = true;
            expr->mInside->mContentDirty = true;
            return true;
        });
        for (auto v : visited) {
            v->setVisited(false);
        }
        dst->mFrom->visitOutputs([src, dst](EXPRP expr, int index) {
            for (int i =0; i< expr->inputs().size(); ++i) {
                auto input = expr->inputs()[i];
                if (input == dst) {
                    expr->mInputs[i] = src;
                }
            }
            src->mFrom->mTo.emplace_back(expr);
            return false;
        });

        dst->mFrom = src->mFrom;
        dst->mFromIndex = src->mFromIndex;
        return;
    }
    Expr::replace(dst->mFrom, src->mFrom);
    dst->mFromIndex = src->mFromIndex;
}

const Variable::Info* Variable::getInfo() {
    if (nullptr == mFrom) {
        return nullptr;
    }
    auto res = mFrom->requireInfo();
    if (!res) {
        return nullptr;
    }
    return mFrom->mInside->mOutputInfos.data() + mFromIndex;
}

bool Variable::resize(INTS dims) {
    if (nullptr != mFrom->get() && VARP::INPUT != mFrom->mType) {
        MNN_ERROR("Can't resize variable not from input\n");
        return false;
    }
    auto& info = mFrom->mInside->mOutputInfos[0];
    if (dims.size() == info.dim.size()) {
        bool theSame = true;
        for (int i=0; i<dims.size(); ++i) {
            if (info.dim[i] != dims[i]) {
                theSame = false;
                break;
            }
        }
        if (theSame) {
            return true;
        }
    }
    info.dim = dims;
    info.syncSize();
    Utils::copyInfoToTensor(mFrom->inside()->mOutputTensors[0], mFrom->inside()->mOutputInfos.data());
    Utils::releaseMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    if (0 >= info.size) {
        return false;
    }
    bool res = Utils::allocMemoryForHostTensor(mFrom->inside()->mOutputTensors[0]);
    if (!res) {
        return false;
    }

    mFrom->mValid = true;
    mFrom->inside()->mInfoDirty = false;
    mFrom->inside()->mContentDirty = true;
    mFrom->visitOutputs([](EXPRP expr, int index) { return expr->setInfoDirty(); });
    return true;
}
void Expr::visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after) {
    bool next = before(expr);
    if (!next) {
        return;
    }
    for (int i = 0; i < expr->inputs().size(); ++i) {
        visit(expr->inputs()[i]->mFrom, before, after);
    }
    after(expr);
}

void* Variable::readInternal(bool forShape, bool swap) {
    if(swap) {
        auto fn = ("swap/" + name()).c_str();
        FILE* f = fopen(fn, "rb");
        if (f != nullptr) {
            // 说明f是存在的
            auto varp = load(fn)[0];
            if(mFrom->mInside->mOutputTensors[mFromIndex]->buffer().host != nullptr) {

            }
            auto info = mFrom->mInside->mOutputInfos[0];
            ::memcpy(mFrom->inside()->mOutputTensors[0]->buffer().host,
                     varp->mFrom->mInside->mOutputTensors[mFromIndex]->buffer().host,
                     info.size * info.type.bytes());
            varp.reset();
            return mFrom->mInside->mOutputTensors[mFromIndex]->buffer().host;
        }
    }
    if (nullptr == mFrom->get()) {
        // Op* mOp, untrainable varp fall into this block
        if (VARP::INPUT == mFrom->mType) {
            if (mFrom->mInside->mContentDirty) {
                return nullptr;
            }
        }
        //MNN_ASSERT(nullptr != mFrom->inside()->mOutputTensors[0]->buffer().host);
        return mFrom->inside()->mOutputTensors[0]->buffer().host;
    }
    auto res = mFrom->requireInfo(); //expr根据自己的input不断调用requireInfo getInfo，递归调用
    if (false == res) {
        return nullptr;
    }
    auto cache = mFrom->inside()->mCache;
    //cache 是每个expr都有的
    if (nullptr == cache) {
        ExecutorScope::Current()->makeCache({mFrom}, forShape);
        cache = mFrom->inside()->mCache;
    }
    if (nullptr == cache) {
        return nullptr;
    }
    if (NO_ERROR != ExecutorScope::Current()->runCache(cache)) {
        return nullptr;
    }
    return Executor::mapOutput(cache.get(), mFrom->mInside->mCacheOffset + mFromIndex, mFrom->mInside->mOutputTensors[mFromIndex]);
}

void Variable::informDirty() {
    std::vector<Expr*> visited;
    mFrom->visitOutputs([&visited](EXPRP expr, int index) {
        if (expr->visited()) {
            return false;
        }
        visited.emplace_back(expr.get());
        expr->setVisited(true);
        if (expr->inside()->mReq.shapeNeedContent.empty()) {
            // Not init
            return false;
        }
        if (expr->inside()->mReq.shapeNeedContent[index]) {
            expr->setInfoDirty();
            expr->visitOutputs([](EXPRP e, int index) { return e->setInfoDirty(); });
            return false;
        }
        if (expr->inside()->mReq.contentNeedContent[index]) {
            if (expr->inside()->mCache != nullptr) {
                Executor::setContentDirty(expr->inside()->mCache.get());
            }
            return true;
        }
        return false;
    });
    for (auto e : visited) {
        e->setVisited(false);
    }
}
void Variable::prepareCompute(const std::vector<VARP>& vars, bool forceCpu) {
    //forceCPU = false
    std::vector<EXPRP> exprs;
    for (auto v : vars) {
        if (!v->expr().first->visited()) {
            assert(v->expr().first->outputSize() == 1);
            v->expr().first->inside()->mCache = nullptr;
            v->expr().first->requireInfo();
            v->expr().first->setVisited(true);
            exprs.emplace_back(v->expr().first);
        }
    }
    for (auto v : vars) {
        v->expr().first->setVisited(false);
    }
    ExecutorScope::Current()->makeCache(std::move(exprs), forceCpu);
}

void* Variable::writeInternal(bool inform) {
    if (nullptr != mFrom->get()) {
        return nullptr;
    }
    if (inform) {
        informDirty();
    }
    mFrom->mInside->mContentDirty = false;
    return mFrom->inside()->mOutputTensors[0]->host<void>();
}

void Variable::unMap() {
    //mFrom->inside()->onUnMapContent(mFromIndex);
}

void Expr::visitOutputs(const std::function<bool(EXPRP, int)>& visit) {
    for (auto iter = mTo.begin(); iter != mTo.end();) {
        auto expr = iter->lock();
        if (nullptr == expr) {
            iter = mTo.erase(iter);
            continue;
        }
        bool recurse = false;
        auto inputs = expr->inputs();
        for (int i=0; i<inputs.size(); ++i) {
            if (inputs[i]->mFrom.get() == this) {
                recurse = recurse || visit(expr, i);
            }
        }
        if (recurse) {
            expr->visitOutputs(visit);
        }
        iter++;
    }
}
bool Expr::setInfoDirty() {
    if (mInside->mInfoDirty && mValid) {
        //MNN_PRINT("End Info Dirty for %s\n", mName.c_str());
        return false;
    }
    //MNN_PRINT("Set Info Dirty for %s\n", mName.c_str());
    mInside->mInfoDirty    = true;
    mInside->mContentDirty = true;
    mValid = true;
    if (mInside->mCache != nullptr) {
        Executor::setShapeDirty(mInside->mCache.get());
    }
    for (auto o : mInside->mOutputTensors) {
        Utils::releaseMemoryForHostTensor(o);
    }
    return true;
}

std::vector<VARP> Variable::load(const char* fileName) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName);
        if (!loader.valid()) {
            MNN_ERROR("Error for open %s\n", fileName);
            return {};
        }
        loader.read();
        if (!loader.valid()) {
            return {};
        }
        loader.merge(buffer);
        if (buffer.get() == nullptr) {
            return {};
        }
    }
    return load(buffer.get(), buffer.size());
}
std::vector<VARP> Variable::load(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    flatbuffers::Verifier verify((const uint8_t*)(buffer), length);
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create variable\n");
        return {};
    }
    std::unique_ptr<NetT> source(UnPackNet(buffer));
    if (nullptr == source) {
        return {};
    }
    if (source->oplists.empty()) {
        MNN_ERROR("Invalid net\n");
        return {};
    }
    // FUNC_PRINT(source->oplists.size());

    auto opSize      = source->oplists.size();
    auto tensorCount = source->tensorName.size();
    if (tensorCount == 0) {
        tensorCount = source->tensorNumber;
    }
    std::vector<VARP> variable;
    variable.reserve(tensorCount);
    std::map<int, VARP> variableMap;

    // Generate All Exprs by order of net
    for (int i = 0; i < opSize; ++i) {
        std::vector<VARP> inputs;
        auto op = source->oplists[i].get();
        for (int index = 0; index < op->inputIndexes.size(); ++index) {
            auto inputIndex = op->inputIndexes[index];
            if (variableMap.find(inputIndex) == variableMap.end()) {
                MNN_ERROR("Can't find variable for %s, the graph is error\n", op->name.c_str());
                break;
            }
            inputs.emplace_back(variableMap[inputIndex]);
        }
        EXPRP expr = Expr::create(source->oplists[i].get(), inputs, (int)op->outputIndexes.size());
        expr->setName(source->oplists[i]->name);

        for (int index = 0; index < op->outputIndexes.size(); ++index) {
            auto outputIndex = op->outputIndexes[index];
            if (variableMap.find(outputIndex) == variableMap.end()) {
                auto newVariable = Variable::create(expr, index);
                if (source->tensorName.size() > outputIndex) {
                    newVariable->setName(source->tensorName[outputIndex]);
                }
                variableMap[outputIndex] = newVariable;
                variable.emplace_back(newVariable);
            }
        }
    }
    return variable;
}

std::map<std::string, VARP> Variable::loadMap(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    auto variables = load(buffer, length);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}

std::map<std::string, VARP> Variable::loadMap(const char* fileName) {
    AUTOTIME;
    auto variables = load(fileName);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}
std::vector<VARP> Variable::mapToSequence(const std::map<std::string, VARP>& source) {
    std::vector<VARP> outputs;
    outputs.reserve(source.size());
    for (auto& iter : source) {
        outputs.emplace_back(iter.second);
    }
    return outputs;
}
void Variable::save(const std::vector<VARP>& vars, NetT* dest, bool swap) {
    std::vector<EXPRP> executeOrder;
    if(swap) {
        for(const auto& v: vars){
            executeOrder.emplace_back(v->mFrom);
        }
    } else {
        executeOrder = getExecuteOrder(vars);
    }

    // Get Expr - TensorOffset Map
    std::map<EXPRP, int> varIndexInfo;
    {
        int tensorOffset = 0;
        for (int i=0; i<executeOrder.size(); ++i) {
            auto expr = executeOrder[i];
            auto outputSize = executeOrder[i]->outputSize();
            varIndexInfo[expr] = tensorOffset;
            tensorOffset += outputSize;
        }
        dest->tensorName.resize(tensorOffset);
    }

    // Create All Op
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto expr = executeOrder[index];
        auto mOp = expr->get();
        std::unique_ptr<OpT> op;
        if (nullptr != mOp) { // untrainable
            op.reset(mOp->UnPack());
        } else { // trainable
            MNN_ASSERT(1 == expr->outputSize());
            auto& info = expr->mInside->mOutputInfos[0];
            auto ptr = expr->mInside->mOutputTensors[0]->host<void>();
            op.reset(new OpT);
            if (expr->mType != VARP::INPUT) {
                auto blob        = new BlobT;
                blob->dataFormat = (MNN_DATA_FORMAT)Utils::convertFormat(info.order);
                blob->dims       = info.dim;
                if (info.type.code == halide_type_float) {
                    blob->dataType = DataType_DT_FLOAT;
                    blob->float32s.resize(info.size);
                    ::memcpy(blob->float32s.data(), ptr, info.size * sizeof(float));
                } else if (info.type.code == halide_type_int && info.type.bits == 32) {
                    blob->dataType = DataType_DT_INT32;
                    blob->int32s.resize(info.size);
                    ::memcpy(blob->int32s.data(), ptr, info.size * sizeof(int));
                } else if (info.type.code == halide_type_int && info.type.bits == 8) {
                    blob->dataType = DataType_DT_INT8;
                    blob->int8s.resize(info.size);
                    auto pptr = (int8_t *)ptr;
                    ::memcpy(blob->int8s.data(), ptr, info.size * sizeof(int8_t));
                } else if (info.type.code == halide_type_uint && info.type.bits == 8) {
                    blob->dataType = DataType_DT_UINT8;
                    blob->uint8s.resize(info.size);
                    ::memcpy(blob->uint8s.data(), ptr, info.size * sizeof(uint8_t));
                }
                op->type       = OpType_Const;
                if (expr->mType == VARP::TRAINABLE) {
                    op->type = OpType_TrainableParam;
                }
                op->main.type  = OpParameter_Blob;
                op->main.value = blob;
            } else {
                op->type                    = OpType_Input;
                op->main.type               = OpParameter_Input;
                op->main.value              = new InputT;
                op->main.AsInput()->dtype   = (MNN::DataType)Utils::convertDataType(info.type);
                MNN_ASSERT(op->main.AsInput()->dtype != DataType_DT_INVALID);
                op->main.AsInput()->dims    = info.dim;
                op->main.AsInput()->dformat = (MNN_DATA_FORMAT)Utils::convertFormat(info.order);
            }
        }
        op->name = expr->name();
        op->inputIndexes.resize(expr->inputs().size());
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            auto inputExpr = expr->inputs()[i]->expr();
            op->inputIndexes[i] = varIndexInfo[inputExpr.first] + inputExpr.second;
        }
        if (op->name.empty()) {
            op->name = EnumNameOpType(op->type) + numberToString(index+1);
        }
        op->outputIndexes.resize(expr->outputSize());
        auto tensorIndexOffset = varIndexInfo[expr];
        for (int v=0; v<expr->outputSize(); ++v) {
            op->outputIndexes[v] = tensorIndexOffset + v;
            dest->tensorName[tensorIndexOffset+v] = expr->outputName(v);
        }
        dest->oplists.emplace_back(std::move(op));
    }

    // Fill Empty Tensor Name With Default Op Name
    for (int index = 0; index < executeOrder.size(); ++index) {
        auto expr = executeOrder[index];
        auto op = dest->oplists[index].get();
        auto tensorIndexOffset = varIndexInfo[expr];
        for (int v=0; v<expr->outputSize(); ++v) {
            auto subindex = tensorIndexOffset + v;
            if (dest->tensorName[subindex].empty()) {
                if (v == 0) {
                    dest->tensorName[subindex] = op->name;
                } else {
                    dest->tensorName[subindex] = op->name + numberToString(v);
                }
            }
        }
    }
}
void Variable::save(const std::vector<VARP>& vars, const char* fileName, bool swap) {
    std::unique_ptr<NetT> net(new NetT);
    save(vars, net.get(), swap);
    // FUNC_PRINT(net->oplists.size());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, net.get());
    builder.Finish(offset);
    // TODO, use FileWriter instead
    FILE* f = fopen(fileName, "wb");
    if (nullptr == f) {
        MNN_ERROR("Open %s error\n", fileName);
        return;
    }
    static const size_t block = 4096;
    size_t totalSize    = builder.GetSize();
    size_t blockSize    = UP_DIV(totalSize, block);
    for (size_t i = 0; i < blockSize; ++i) {
        size_t sta = block * i;
        size_t fin = std::min(sta + block, totalSize);
        if (fin > sta) {
            auto realSize = fwrite((const char*)builder.GetBufferPointer() + sta, 1, fin - sta, f);
            if (realSize != fin - sta) {
                MNN_ERROR("Write %s error\n", fileName);
            }
        }
    }
    fclose(f);
}
std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> Variable::getInputAndOutput(const std::map<std::string, VARP>& allVariable) {
    std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> res;
    for (auto& iter : allVariable) {
        auto var = iter.second;
        if (var->expr().first->get() == nullptr && var->expr().first->mType == VARP::INPUT) {
            res.first[var->name()] = var;
        }
        if (var->linkNumber() == 0) {
            res.second[var->name()] = var;
        }
    }
    return res;
}

std::vector<EXPRP> Variable::getExecuteOrder(const std::vector<VARP>& outputs) {
    std::vector<EXPRP> sequence;
    for (auto output : outputs) {
        Expr::visit(
                        output->mFrom, [](EXPRP expr) { return !expr->visited(); },
                        [&sequence](EXPRP expr) {
                            //FUNC_PRINT_ALL(var->name().c_str(), s);
                            if (!expr->visited()) {
                                sequence.emplace_back(expr);
                                expr->setVisited(true);
                            }
                            return true;
                        });
    }
    for (auto expr : sequence) {
        expr->setVisited(false);
    }
    return sequence;
}

VARP VARP::operator+(VARP var) const {
    return _Add(VARP(mContent), var);
}
VARP VARP::operator-(VARP var) const {
    return _Subtract(VARP(mContent), var);
}
VARP VARP::operator*(VARP var) const {
    return _Multiply(VARP(mContent), var);
}
VARP VARP::operator/(VARP var) const {
    return _Divide(VARP(mContent), var);
}
VARP VARP::mean(INTS dims) const {
    return _ReduceMean(VARP(mContent), dims);
}
VARP VARP::sum(INTS dims) const {
    return _ReduceSum(VARP(mContent), dims);
}

} // namespace Express
} // namespace MNN
