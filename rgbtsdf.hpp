/**
 * Copyright (c) 2023, Li Yunqiang, walkfish8@hotmail.com.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the organization nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTOR BE
 * LIABLE  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#ifndef _RGBTSDF_RGBTSDF_HPP_
#define _RGBTSDF_RGBTSDF_HPP_

#define RGBTSDF_VERSION_MAJOR 1
#define RGBTSDF_VERSION_MINOR 2
#define RGBTSDF_VERSION_PATCH 0

#include <set>
#include <mutex>
#include <cmath>
#include <stack>
#include <future>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <numeric>
#include <functional>
#include <unordered_map>

/** Using OpenMP To Accelerate Program. */
#ifdef RGBTSDF_USE_OMP
#include <omp.h>
#endif

#ifdef RGBTSDF_USE_EIGEN
#include <Eigen/Core>
#include <Eigen/Sparse>
#endif

/** Print Detailed Information. */
#ifdef RGBTSDF_PRINT_VERBOSE
#define RGBTSDF_PRINT(str, ...) \
    printf("[RGBTSDF INFO] "), printf(str, ##__VA_ARGS__), printf("\n")
#else
#define RGBTSDF_PRINT(str, ...) ((void)0)
#endif

/** Consider The Tranpose Matrix of R As Inverse Matrix. */
#define RGBTSDF_USE_FAST_RT_INV
/** The Truncated Distance, Which Means The Times of Unit Size. */
#ifndef RGBTSDF_TRUNCATED_FACTOR
#define RGBTSDF_TRUNCATED_FACTOR 8
#endif
/** define helper macro pi to calculate angle. */
#define RGBTSDF_PI 3.1415926535897932384626433832795

namespace rgbtsdf
{
#ifndef __GNUC__
typedef int          int32_t;
typedef long long    int64_t;
typedef unsigned int uint32_t;
#else
typedef int __attribute__((__may_alias__))          int32_t;
typedef long long __attribute__((__may_alias__))    int64_t;
typedef unsigned int __attribute__((__may_alias__)) uint32_t;
#endif

#ifdef RGBTSDF_USE_OMP
constexpr static bool defaultMultiThreadFlag = true;
#else
constexpr static bool defaultMultiThreadFlag = false;
#endif

// clang-format off
template <typename _Tp> struct CoordTrait { typedef _Tp type; };
template <typename _Tp> using coord_trait_t = typename CoordTrait<_Tp>::type;

// check cvmat type
template <typename T, typename = int> struct has_ptr: std::false_type {};
template <typename T>
struct has_ptr<T, decltype((void)std::declval<T>().ptr(0), 0)>: std::true_type {};
template <typename T, typename = int> struct has_step: std::false_type {};
template <typename T>
struct has_step<T, decltype((void)std::declval<T>().step, 0)>: std::true_type {};
template <typename T, typename = int> struct has_rows: std::false_type {};
template <typename T>
struct has_rows<T, decltype((void)std::declval<T>().rows, 0)>: std::true_type {};
template <typename T, typename = int> struct has_cols: std::false_type {};
template <typename T>
struct has_cols<T, decltype((void)std::declval<T>().cols, 0)>: std::true_type {};
template <typename T, typename = void> struct is_cvmat 
{ enum { value = has_ptr<T>::value & has_step<T>::value & has_rows<T>::value & has_cols<T>::value }; };

#define SAMESIGN(a,b) (((a == 0) == (b == 0)) && ((a < 0) == (b < 0)))
template <typename _Tp> static inline bool NotSameSign(_Tp a, _Tp b) { return !SAMESIGN(a, b); }
static inline bool NotSameSign(float a, float b)
{ return ((*(int32_t*)&a) & 0x80000000) ^ ((*(int32_t*)&b) & 0x80000000); }
static inline bool NotSameSign(double a, double b)
{ return ((*(int64_t*)&a) & 0x8000000000000000) ^ ((*(int64_t*)&b) & 0x8000000000000000); }

template <int _N> struct BinaryExponent {
    static_assert(_N >= 0, "_N Should Be Positive Number.");
    enum { value = BinaryExponent<_N - 1>::value << 1 };
};
template <> struct BinaryExponent<0> { enum { value = 1 }; };
/** Get How Many Bits To Represent Number. */
template <int _N> struct MaxBinaryExponent {
    static_assert(_N >= 0, "_N Should Be Positive Number.");
    enum { value = MaxBinaryExponent<_N / 2>::value + 1 };
};
template <> struct MaxBinaryExponent<0> { enum { value = 0 }; };

template <typename _Tp> static inline _Tp clamp(_Tp val, _Tp minVal, _Tp maxVal)
{ return (std::max)((std::min)(val, maxVal), minVal); }

template <typename _Tp> struct Size_ {
    Size_()             = default;
    Size_(const Size_&) = default;
    Size_(_Tp width, _Tp height) noexcept : width(width), height(height) {}
    inline _Tp count() const { return width * height; }
    inline bool operator==(const Size_<_Tp>& sz) const noexcept
    { return sz.width == width && sz.height == height; }
    inline bool operator!=(const Size_<_Tp>& sz) const noexcept
    { return sz.width != width || sz.height != height; }
    _Tp width, height;
};
template <typename _Tp> struct CoordTrait<Size_<_Tp>> { typedef _Tp type; };

struct RGBPixel {
    RGBPixel()                = default;
    RGBPixel(const RGBPixel&) = default;
    RGBPixel(unsigned char r, unsigned char g, unsigned char b) noexcept : r(r), g(g), b(b)
    {}
    template <typename T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    RGBPixel(T r, T g, T b) noexcept : r((unsigned char)r), g((unsigned char)g), b((unsigned char)b)
    {}
    unsigned char r, g, b; /* same as opencv BGR sequence. */
};
template <> struct CoordTrait<RGBPixel> { typedef unsigned char type; };

template <typename _Tp = float> struct Point3_ {
    Point3_()               = default;
    Point3_(const Point3_&) = default;
    Point3_(_Tp x, _Tp y, _Tp z) noexcept : x(x), y(y), z(z)
    {}
    template <typename T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Point3_(T x, T y, T z) noexcept : x(_Tp(x)), y(_Tp(y)), z(_Tp(z))
    {}
    template <typename T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Point3_(const T data[3]) noexcept : x((_Tp)data[0]), y((_Tp)data[1]), z((_Tp)data[2])
    {}
    _Tp dot(const Point3_& p) const noexcept { return x*p.x+y*p.y+z*p.z; }
    double ddot(const Point3_& p) const noexcept { return (double)x*p.x+(double)y*p.y+(double)z*p.z; }
    Point3_& normalize() noexcept
    { auto norm=(std::sqrt)(x*x+y*y+z*z); if(norm) x/=norm, y/=norm, z/=norm; return *this; }
    double norm() const noexcept { return (std::sqrt)((double)x*x + (double)y*y + (double)z*z); }
    double norm2() const noexcept { return (double)x*x + (double)y*y + (double)z*z; }

    bool operator>(_Tp v) const { return x > v && y > v && z > v; }
    bool operator<(_Tp v) const { return x < v && y < v && z < v; }
    bool operator!=(_Tp v) const { return x != v && y != v && z != v; }
    bool operator>=(_Tp v) const { return x >= v && y >= v && z >= v; }
    bool operator<=(_Tp v) const { return x <= v && y <= v && z <= v; }
    Point3_ operator-(void) const { return {-x, -y, -z}; }
    Point3_ operator+(_Tp v) const { return {x + v, y + v, z + v}; }
    Point3_ operator-(_Tp v) const { return {x - v, y - v, z - v}; }
    Point3_ operator*(_Tp v) const { return {x * v, y * v, z * v}; }
    Point3_ operator/(_Tp v) const { return {x / v, y / v, z / v}; }
    Point3_& operator+=(_Tp v) { x += v, y += v, z += v; return *this; }
    Point3_& operator-=(_Tp v) { x -= v, y -= v, z -= v; return *this; }
    Point3_& operator*=(_Tp v) { x *= v, y *= v, z *= v; return *this; }
    Point3_& operator/=(_Tp v) { x /= v, y /= v, z /= v; return *this; }
    Point3_ operator+(const Point3_& p) const { return {x + p.x, y + p.y, z + p.z}; }
    Point3_ operator-(const Point3_& p) const { return {x - p.x, y - p.y, z - p.z}; }
    Point3_ operator*(const Point3_& p) const { return {x * p.x, y * p.y, z * p.z}; }
    Point3_ operator/(const Point3_& p) const { return {x / p.x, y / p.y, z / p.z}; }
    Point3_& operator+=(const Point3_& p) { x += p.x, y += p.y, z += p.z; return *this; }
    Point3_& operator-=(const Point3_& p) { x -= p.x, y -= p.y, z -= p.z; return *this; }
    Point3_& operator*=(const Point3_& p) { x *= p.x, y *= p.y, z *= p.z; return *this; }
    Point3_& operator/=(const Point3_& p) { x /= p.x, y /= p.y, z /= p.z; return *this; }

    _Tp x, y, z;
};
template <typename _Tp> struct CoordTrait<Point3_<_Tp>> { typedef _Tp type; };
template <typename _Tp> static inline Point3_<_Tp> operator+(
    coord_trait_t<_Tp> v, const Point3_<_Tp>& p) { return {v + p.x, v + p.y, v + p.z}; }
template <typename _Tp> static inline Point3_<_Tp> operator-(
    coord_trait_t<_Tp> v, const Point3_<_Tp>& p) { return {v - p.x, v - p.y, v - p.z}; }
template <typename _Tp> static inline Point3_<_Tp> operator*(
    coord_trait_t<_Tp> v, const Point3_<_Tp>& p) { return {v * p.x, v * p.y, v * p.z}; }
template <typename _Tp> static inline Point3_<_Tp> operator/(
    coord_trait_t<_Tp> v, const Point3_<_Tp>& p) { return {v / p.x, v / p.y, v / p.z}; }
// clang-format on

enum class TSDFType : int { TSDF = 0, TINYTSDF, RGBTSDF };

template <TSDFType _Type, typename _Float = float> struct TSDFVoxel_ {
    TSDFVoxel_()                  = default;
    TSDFVoxel_(const TSDFVoxel_&) = default;
    inline TSDFVoxel_& operator+=(const TSDFVoxel_& vox) noexcept
    {
        w += vox.w, v += (vox.v - v) * vox.w / w;
        return *this;
    }
    inline TSDFVoxel_& operator-=(const TSDFVoxel_& vox) noexcept
    {
        w -= vox.w, v += (v - vox.v) * vox.w / w;
        return *this;
    }
    inline int    time() const noexcept { return 0; }
    inline _Float value() const noexcept { return v; }
    inline _Float weight() const noexcept { return w; }

    _Float v, w;
};
template <typename _Float> struct TSDFVoxel_<TSDFType::RGBTSDF, _Float> {
    TSDFVoxel_()                  = default;
    TSDFVoxel_(const TSDFVoxel_&) = default;
    inline TSDFVoxel_& operator+=(const TSDFVoxel_& vox) noexcept
    {
        w += vox.w;
        _Float wd = vox.w / w;
        v += (vox.v - v) * wd;
        r = (unsigned short)clamp<_Float>(
            r + ((_Float)vox.r - r) * wd, 0x0, 0xFFFF);
        g = (unsigned short)clamp<_Float>(
            g + ((_Float)vox.g - g) * wd, 0x0, 0xFFFF);
        b = (unsigned short)clamp<_Float>(
            b + ((_Float)vox.b - b) * wd, 0x0, 0xFFFF);
        return *this;
    }
    inline TSDFVoxel_& operator-=(const TSDFVoxel_& vox) noexcept
    {
        w -= vox.w;
        _Float wd = vox.w / w;
        v += (v - vox.v) * wd;
        r = (unsigned short)clamp<_Float>(
            r + (r - (_Float)vox.r) * wd, 0x0, 0xFFFF);
        g = (unsigned short)clamp<_Float>(
            g + (g - (_Float)vox.g) * wd, 0x0, 0xFFFF);
        b = (unsigned short)clamp<_Float>(
            b + (b - (_Float)vox.b) * wd, 0x0, 0xFFFF);
        return *this;
    }
    inline int    time() const noexcept { return t; }
    inline _Float value() const noexcept { return v; }
    inline _Float weight() const noexcept { return w; }

    _Float         v, w;
    unsigned short r, g, b, t;
};
template <typename _Float> struct TSDFVoxel_<TSDFType::TINYTSDF, _Float> {
    TSDFVoxel_()                  = default;
    TSDFVoxel_(const TSDFVoxel_&) = default;
    inline TSDFVoxel_& operator+=(const TSDFVoxel_& vox) noexcept
    {
        float        vv = value();
        float        vn = vox.value();
        unsigned int wn = (unsigned int)e + vox.e;
        vv += (vn - vv) * vox.e / wn;
        e        = wn < 0xFF ? wn : 0xFF;
        float ss = vv;
        (*(uint32_t*)& ss &= 0x80000000) |= 0x3F800000U;
        vv += ss;
        (*(uint32_t*)this &= 0x7F800000) |= *(uint32_t*)&vv & 0x807FFFFF;
    }
    inline TSDFVoxel_& operator-=(const TSDFVoxel_& vox) noexcept
    {
        float vv = value();
        float vn = vox.value();
        int   wn = (int)e - vox.e;
        vv += (vv - vn) * vox.e / wn;
        e        = wn <= 0x0 ? 0x0 : (unsigned int)wn;
        float ss = vv;
        (*(uint32_t*)& ss &= 0x80000000) |= 0x3F800000U;
        vv += ss;
        (*(uint32_t*)this &= 0x7F800000) |= *(uint32_t*)&vv & 0x807FFFFF;
    }
    inline int    time() const noexcept { return 0; }
    inline _Float value() const noexcept
    {
        float vv = *reinterpret_cast<const float*>(this);
        (*(uint32_t*)& vv &= 0x807FFFFF) |= 0x3F800000U;
        float ss = *reinterpret_cast<const float*>(this);
        (*(uint32_t*)& ss &= 0x80000000) |= 0x3F800000U;
        return vv - ss;
    }
    inline _Float weight() const noexcept { return static_cast<_Float>(e); }

    unsigned int m : 23;
    unsigned int e : 8;
    unsigned int s : 1;
};

/** uncomment other implement, show error when not define default value. */
template <TSDFType _Type, typename _Float> struct TSDFDefaultValue {
    const static TSDFVoxel_<_Type, _Float> value;
};
template <> const TSDFVoxel_<TSDFType::TSDF, float>
    TSDFDefaultValue<TSDFType::TSDF, float>::value = {1, 0};
template <> const TSDFVoxel_<TSDFType::TSDF, double>
    TSDFDefaultValue<TSDFType::TSDF, double>::value = {1, 0};
template <> const TSDFVoxel_<TSDFType::TINYTSDF, float>
    TSDFDefaultValue<TSDFType::TINYTSDF, float>::value = {0, 0, 0};
template <> const TSDFVoxel_<TSDFType::TINYTSDF, double>
    TSDFDefaultValue<TSDFType::TINYTSDF, double>::value = {0, 0, 0};
template <> const TSDFVoxel_<TSDFType::RGBTSDF, float>
    TSDFDefaultValue<TSDFType::RGBTSDF, float>::value = {1, 0, 0, 0, 0, 0};
template <> const TSDFVoxel_<TSDFType::RGBTSDF, double>
    TSDFDefaultValue<TSDFType::RGBTSDF, double>::value = {1, 0, 0, 0, 0, 0};

template <TSDFType _Type, typename _Float, bool _Inverse>
struct TSDFValueSetter {
    static inline void add(
        TSDFVoxel_<_Type, _Float>& voxel, _Float vn, _Float wn, ...) noexcept
    {
        voxel.w += wn, voxel.v += (vn - voxel.v) * wn / voxel.w;
    }
};
template <TSDFType _Type, typename _Float>
struct TSDFValueSetter<_Type, _Float, true> {
    static inline void add(
        TSDFVoxel_<_Type, _Float>& voxel, _Float vn, _Float wn, ...) noexcept
    {
        voxel.w -= wn, voxel.v += (voxel.v - vn) * wn / voxel.w;
    }
};
template <typename _Float>
struct TSDFValueSetter<TSDFType::RGBTSDF, _Float, false> {
    static inline void add(TSDFVoxel_<TSDFType::RGBTSDF, _Float>& voxel,
        _Float vn, _Float wn, const RGBPixel& rgb, int curTime) noexcept
    {
        voxel.w += wn;
        _Float wd = wn / voxel.w;
        voxel.v += (vn - voxel.v) * wd;
        voxel.r = (unsigned short)clamp<_Float>(
            voxel.r + (_Float)((rgb.r << 0x8) - voxel.r) * wd, 0x0, 0xFFFF);
        voxel.g = (unsigned short)clamp<_Float>(
            voxel.g + (_Float)((rgb.g << 0x8) - voxel.g) * wd, 0x0, 0xFFFF);
        voxel.b = (unsigned short)clamp<_Float>(
            voxel.b + (_Float)((rgb.b << 0x8) - voxel.b) * wd, 0x0, 0xFFFF);
        voxel.t = static_cast<unsigned short>(curTime);
    }
};
template <typename _Float>
struct TSDFValueSetter<TSDFType::RGBTSDF, _Float, true> {
    static inline void add(TSDFVoxel_<TSDFType::RGBTSDF, _Float>& voxel,
        _Float vn, _Float wn, const RGBPixel& rgb, ...) noexcept
    {
        voxel.w -= wn;
        _Float wd = wn / voxel.w;
        voxel.v += (voxel.v - vn) * wd;
        voxel.r = (unsigned short)clamp<_Float>(
            voxel.r + (_Float)(voxel.r - (rgb.r << 0x8)) * wd, 0x0, 0xFFFF);
        voxel.g = (unsigned short)clamp<_Float>(
            voxel.g + (_Float)(voxel.g - (rgb.g << 0x8)) * wd, 0x0, 0xFFFF);
        voxel.b = (unsigned short)clamp<_Float>(
            voxel.b + (_Float)(voxel.b - (rgb.b << 0x8)) * wd, 0x0, 0xFFFF);
    }
};
template <typename _Float>
struct TSDFValueSetter<TSDFType::TINYTSDF, _Float, false> {
    static inline void add(TSDFVoxel_<TSDFType::TINYTSDF, _Float>& voxel,
        _Float vn, _Float /*wn*/, ...) noexcept
    {
        // if (wn < 0.9) return;
        float vv = voxel.value();
        if (voxel.e != 0xFF) vv += (vn - vv) / ++voxel.e;
        else
            vv += (vn - vv) / 0x100;
        float ss = vv;
        (*(uint32_t*)(&ss) &= 0x80000000) |= 0x3F800000U;
        vv += ss;
        (*(uint32_t*)(&voxel) &= 0x7F800000) |= (*(uint32_t*)&vv & 0x807FFFFF);
    }
};
template <typename _Float>
struct TSDFValueSetter<TSDFType::TINYTSDF, _Float, true> {
    static inline void add(TSDFVoxel_<TSDFType::TINYTSDF, _Float>& voxel,
        _Float vn, _Float /*wn*/, ...) noexcept
    {
        // if (wn < 0.9) return;
        float vv = voxel.value();
        if (voxel.e > 0x1) vv += (vv - vn) / --voxel.e;
        else
            vv = 1, voxel.e = 0;
        float ss = vv;
        (*(uint32_t*)(&ss) &= 0x80000000) |= 0x3F800000U;
        vv += ss;
        (*(uint32_t*)(&voxel) &= 0x7F800000) |= (*(uint32_t*)&vv & 0x807FFFFF);
    }
};

template <TSDFType _Type, typename _Float> struct TSDFPixelGetter {
    static inline void get(...) noexcept {}
    static inline bool check(...) noexcept { return true; }
    static inline void interp(...) noexcept {}
};
template <typename _Float> struct TSDFPixelGetter<TSDFType::RGBTSDF, _Float> {
    static inline void get(const TSDFVoxel_<TSDFType::RGBTSDF, _Float>* vox,
        RGBPixel* rgb) noexcept
    {
        assert(vox);
        if (!rgb) return;
        *rgb = {vox->r >> 0x8, vox->g >> 0x8, vox->b >> 0x8};
    }
    static inline void get(const TSDFVoxel_<TSDFType::RGBTSDF, _Float>* vox,
        Point3_<_Float>* rgb) noexcept
    {
        assert(vox);
        if (!rgb) return;
        *rgb = {vox->r / 65535.0f, vox->g / 65535.0f, vox->b / 65535.0f};
    }
    static inline bool check(bool hasColor) noexcept { return hasColor; }
    static inline void interp(const TSDFVoxel_<TSDFType::RGBTSDF, _Float>* cur,
        const TSDFVoxel_<TSDFType::RGBTSDF, _Float>*                       next,
        std::vector<RGBPixel>* pixels) noexcept
    {
        _Float curVal  = (std::abs)(cur->v);
        _Float nextVal = (std::abs)(next->v);
        _Float sumVal  = (curVal + nextVal) * 0x100;
        pixels->emplace_back(
            (unsigned char)clamp<_Float>(
                (cur->r * nextVal + next->r * curVal) / sumVal, 0x0, 0xFF),
            (unsigned char)clamp<_Float>(
                (cur->g * nextVal + next->g * curVal) / sumVal, 0x0, 0xFF),
            (unsigned char)clamp<_Float>(
                (cur->b * nextVal + next->b * curVal) / sumVal, 0x0, 0xFF));
    }
};

struct noArr {};
template <typename _Tp> struct OutArr_;
template <typename _Tp> struct CountNonZeroImpl;
template <typename _Tp, typename _Float> struct RowConvolveImpl;
template <typename _Tp, typename _Float> struct ColConvolveImpl;

template <typename _Tp> struct InArr_ {
    constexpr static int _ElementSize = (int)sizeof(_Tp);
    InArr_()                          = delete;
    InArr_(const InArr_&)             = default;
    InArr_(const noArr&) : data(nullptr), stride(0), width(0), height(0) {}
    template <int _N> InArr_(const _Tp (&arr)[_N])
        : data(arr), stride((int)sizeof(_Tp[_N])), width(_N), height(1)
    {}
    template <int _R, int _C> InArr_(const _Tp (&arr)[_R][_C])
        : data((const _Tp*)arr)
        , stride((int)sizeof(_Tp[_C]))
        , width(_C)
        , height(_R)
    {}
    InArr_(const std::tuple<const _Tp*, int, int>& tuple)
        : data(std::get<0>(tuple))
        , stride(_ElementSize * std::get<1>(tuple))
        , width(std::get<1>(tuple))
        , height(std::get<2>(tuple))
    {}
    InArr_(const void* data, int num)
        : data((const _Tp*)data)
        , stride(_ElementSize * num)
        , width(num)
        , height(1)
    {}
    template <typename _CVMAT,
        class = typename std::enable_if<is_cvmat<_CVMAT>::value>::type>
    InArr_(const _CVMAT& mat)
        : data(mat.template ptr<_Tp>())
        , stride((int)mat.step)
        , width(mat.cols)
        , height(mat.rows)
    {}
    InArr_(const std::vector<_Tp>& vec)
        : data(vec.data())
        , stride((int)(vec.size() * sizeof(_Tp)))
        , width((int)vec.size())
        , height(1)
    {}
    InArr_(const void* data, Size_<int> sz)
        : data((const _Tp*)data)
        , stride(sz.width * _ElementSize)
        , width(sz.width)
        , height(sz.height)
    {}
    InArr_(const void* data, int width, int height)
        : data((const _Tp*)data)
        , stride(width * _ElementSize)
        , width(width)
        , height(height)
    {}
    InArr_(const void* data, int stride, int width, int height)
        : data((const _Tp*)data), stride(stride), width(width), height(height)
    {}
    Size_<int> size() const
    {
        return data ? Size_<int>(width, height) : Size_<int>(0, 0);
    }
    int count() const noexcept { return !empty() ? width * height : 0; }
    // return pointer of data offset
    const _Tp* ptr(int ind = 0) const
    {
        assert(ind >= 0 && ind < height);
        if (stride <= 0) return data + (size_t)ind * width;
        return (const _Tp*)((const char*)data + (size_t)ind * stride);
    }
    bool empty() const { return !(data && width > 0 && height > 0); }
    bool isMatrix() const { return width > 1 && height > 1; }
    bool isVector() const { return width == 1 || height == 1; }
    bool isContinuous() const { return stride == width * _ElementSize; }
    void copyTo(OutArr_<_Tp> arr) const
    {
        assert(!empty() && !arr.empty() && size() == arr.size());
        for (int y = 0; y < height; ++y)
            memcpy(arr.ptr(y), ptr(y), width * sizeof(_Tp));
    }
    template <typename T,
        class = typename std::enable_if<!std::is_same<T, _Tp>::value>::type>
    void copyTo(OutArr_<T> arr) const
    {
        assert(!empty() && !arr.emtpy() && size() == arr.size());
        for (int y = 0; y < height; ++y) {
            const _Tp* src = ptr(y);
            T*         dst = arr.ptr(y);
            for (int x = 0; x < width; ++x) dst[x] = src[x];
        }
    }
    void copyTo(std::vector<_Tp>& vec, InArr_<unsigned char> mask) const
    {
        assert(!empty() && !mask.empty() && size() == mask.size());
        int cnt = mask.countNonZero();
        if (!cnt) return;
        vec.resize(cnt), cnt = 0;
        for (int y = 0; y < height; ++y) {
            const _Tp*           valptr  = ptr(y);
            const unsigned char* maskptr = mask.ptr(y);
            for (int x = 0; x < width; ++x)
                if (maskptr[x]) vec[cnt++] = valptr[x];
        }
    }
    int countNonZero() const
    {
        if (empty()) return 0;
        if (isContinuous()) return CountNonZeroImpl<_Tp>::proc(ptr(), count());
        int sum = 0;
        for (int y = 0; y < height; ++y)
            sum += CountNonZeroImpl<_Tp>::proc(ptr(y), width);
        return sum;
    }
    const _Tp* data;
    const int  stride, width, height;
};

template <typename _Tp> struct OutArr_: InArr_<_Tp> {
    OutArr_()               = delete;
    OutArr_(const OutArr_&) = default;
    OutArr_(const noArr&) : InArr_<_Tp>(noArr()) {}
    template <int _N> OutArr_(const _Tp (&arr)[_N]) : InArr_<_Tp>(arr) {}
    template <int _R, int _C> OutArr_(const _Tp (&arr)[_R][_C])
        : InArr_<_Tp>(arr)
    {}
    OutArr_(void* data, int num) : InArr_<_Tp>(data, num) {}
    template <typename _CVMAT,
        class = typename std::enable_if<is_cvmat<_CVMAT>::value>::type>
    OutArr_(const _CVMAT& mat) : InArr_<_Tp>(mat)
    {}
    OutArr_(const void* data, Size_<int> sz) : InArr_<_Tp>(data, sz) {}
    OutArr_(void* data, int width, int height)
        : InArr_<_Tp>(data, width, height)
    {}
    OutArr_(void* data, int stride, int width, int height)
        : InArr_<_Tp>(data, stride, width, height)
    {}
    _Tp* ptr(int ind = 0) { return const_cast<_Tp*>(InArr_<_Tp>::ptr(ind)); }
    void setZero()
    {
        if (InArr_<_Tp>::empty()) return;
        for (int y = 0; y < InArr_<_Tp>::height; ++y)
            memset(ptr(y), 0, InArr_<_Tp>::width * sizeof(_Tp));
    }
    void setNegOne()
    {
        if (InArr_<_Tp>::empty()) return;
        for (int y = 0; y < InArr_<_Tp>::height; ++y)
            memset(ptr(y), -1, InArr_<_Tp>::width * sizeof(_Tp));
    }
    void copyFrom(InArr_<_Tp> arr)
    {
        if (!arr.empty() && arr.size() == InArr_<_Tp>::size())
            arr.copyTo(*this);
    }
    template <typename T,
        class = typename std::enable_if<!std::is_same<T, _Tp>::value>::type>
    void copyFrom(InArr_<T> arr)
    {
        if (!arr.empty() && arr.size() == InArr_<_Tp>::size())
            arr.copyTo(*this);
    }
};

template <typename _Tp> struct CountNonZeroImpl {
    static inline int proc(const _Tp* in, int num)
    {
        int cnt = 0;
        for (const _Tp* cur = in; cur != in + num; ++cur)
            if (*cur != 0) cnt++;
        return cnt;
    }
};

template <typename _Float = double> struct K_ {
    static_assert(
        std::is_floating_point<_Float>::value, "Should be float or double.");
    K_()          = delete;
    K_(const K_&) = default;
    template <typename _Tp> K_(const K_<_Tp>& K)
        : fx((_Float)K.fx), fy((_Float)K.fy), cx((_Float)K.cx), cy((_Float)K.cy)
    {}
    template <typename _CVMAT,
        class = typename std::enable_if<is_cvmat<_CVMAT>::value>::type>
    K_(const _CVMAT& mat)
        : K_<_Float>(mat.type() == 6 ?
                         K_<_Float>(mat.template ptr<double[3]>()) :
                         K_<_Float>(mat.template ptr<float[3]>()))
    {}
    K_(const InArr_<float>& arr) : K_<_Float>(arr.ptr()) {}
    K_(const InArr_<double>& arr) : K_<_Float>(arr.ptr()) {}
    template <typename _Tp,
        class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    K_(_Tp f, _Tp cx, _Tp cy)
        : fx((_Float)f), fy((_Float)f), cx((_Float)cx), cy((_Float)cy)
    {}
    template <typename _Tp,
        class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    K_(_Tp fx, _Tp fy, _Tp cx, _Tp cy)
        : fx((_Float)fx), fy((_Float)fy), cx((_Float)cx), cy((_Float)cy)
    {}
    template <typename _Tp,
        class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    K_(const _Tp data[4])
        : fx((_Float)data[0])
        , fy((_Float)data[1])
        , cx((_Float)data[2])
        , cy((_Float)data[3])
    {}
    template <typename _Tp,
        class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    K_(const _Tp data[3][3])
        : fx((_Float)data[0][0])
        , fy((_Float)data[1][1])
        , cx((_Float)data[0][2])
        , cy((_Float)data[1][2])
    {}
    _Float fx, fy, cx, cy;
};

template <typename _Float> struct RotateImpl;
template <typename _Float> struct TransformImpl;

template <typename _Float = double> struct RT_ {
    static_assert(std::is_floating_point<_Float>::value,
        "Consider using float or double.");

    RT_() = default;
    template <typename _Tp> RT_(const RT_<_Tp>& RT)
    {
        a1 = (_Float)RT.a1, a2 = (_Float)RT.a2, a3 = (_Float)RT.a3;
        b1 = (_Float)RT.b1, b2 = (_Float)RT.b2, b3 = (_Float)RT.b3;
        c1 = (_Float)RT.c1, c2 = (_Float)RT.c2, c3 = (_Float)RT.c3;
        tx = (_Float)RT.tx, ty = (_Float)RT.ty, tz = (_Float)RT.tz;
    }
    // clang-format off
    template <typename _CVMAT,  class= typename std::enable_if<is_cvmat<_CVMAT>::value>::type>
    RT_(const _CVMAT& mat)
        : RT_<_Float>(mat.type() == 6 ? RT_<_Float>(mat.template ptr<double>()) : RT_<_Float>(mat.template ptr<float>()))
    {}
    RT_(const InArr_<float>& arr) : RT_<_Float>(arr.ptr()) {}
    RT_(const InArr_<double>& arr) : RT_<_Float>(arr.ptr()) {}
    template <typename _Tp, class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    RT_(const _Tp rt[12])
        : a1((_Float)rt[0]), a2((_Float)rt[1]), a3((_Float)rt[2]), tx((_Float)rt[3])
        , b1((_Float)rt[4]), b2((_Float)rt[5]), b3((_Float)rt[6]), ty((_Float)rt[7])
        , c1((_Float)rt[8]), c2((_Float)rt[9]), c3((_Float)rt[10]), tz((_Float)rt[11])
    {}
    template <typename _Tp, class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    RT_(const _Tp rt[3][4])
        : a1((_Float)rt[0][0]), a2((_Float)rt[0][1]), a3((_Float)rt[0][2]), tx((_Float)rt[0][3])
        , b1((_Float)rt[1][0]), b2((_Float)rt[1][1]), b3((_Float)rt[1][2]), ty((_Float)rt[1][3])
        , c1((_Float)rt[2][0]), c2((_Float)rt[2][1]), c3((_Float)rt[2][2]), tz((_Float)rt[2][3])
    {}
    template <typename _Tp, class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    RT_(_Tp a1, _Tp a2, _Tp a3, _Tp b1, _Tp b2, _Tp b3, _Tp c1, _Tp c2, _Tp c3, _Tp tx, _Tp ty, _Tp tz)
        : a1((_Float)a1), a2((_Float)a2), a3((_Float)a3), tx((_Float)tx)
        , b1((_Float)b1), b2((_Float)b2), b3((_Float)b3), ty((_Float)ty)
        , c1((_Float)c1), c2((_Float)c2), c3((_Float)c3), tz((_Float)tz)
    {}
    static const RT_<_Float>& eye() noexcept
    { static RT_<_Float> Identity{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}; return Identity; }
    // clang-format on
    RT_ inv() const
    {
#ifndef RGBTSDF_USE_FAST_RT_INV
        double det = a1 * (b2 * c3 - b3 * c2) + b1 * (c2 * a3 - a2 * c3) +
                     c1 * (a2 * b3 - b2 * a3);
        double A11 = (b2 * c3 - b3 * c2) / det;
        double A12 = (b3 * c1 - b1 * c3) / det;
        double A13 = (b1 * c2 - b2 * c1) / det;
        double A21 = (a3 * c2 - a2 * c3) / det;
        double A22 = (a1 * c3 - a3 * c1) / det;
        double A23 = (a2 * c1 - a1 * c2) / det;
        double A31 = (a2 * b3 - a3 * b2) / det;
        double A32 = (a3 * b1 - a1 * b3) / det;
        double A33 = (a1 * b2 - a2 * b1) / det;
        return {A11, A21, A31, A12, A22, A32, A13, A23, A33,
            -(A11 * tx + A21 * ty + A31 * tz),
            -(A12 * tx + A22 * ty + A32 * tz),
            -(A13 * tx + A23 * ty + A33 * tz)};
#else
        return {a1, b1, c1, a2, b2, c2, a3, b3, c3,
            -(a1 * tx + b1 * ty + c1 * tz), -(a2 * tx + b2 * ty + c2 * tz),
            -(a3 * tx + b3 * ty + c3 * tz)};
#endif
    }
    RT_ dot(const RT_& RT) const
    {
        return {a1 * RT.a1 + a2 * RT.b1 + a3 * RT.c1,
            a1 * RT.a2 + a2 * RT.b2 + a3 * RT.c2,
            a1 * RT.a3 + a2 * RT.b3 + a3 * RT.c3,
            b1 * RT.a1 + b2 * RT.b1 + b3 * RT.c1,
            b1 * RT.a2 + b2 * RT.b2 + b3 * RT.c2,
            b1 * RT.a3 + b2 * RT.b3 + b3 * RT.c3,
            c1 * RT.a1 + c2 * RT.b1 + c3 * RT.c1,
            c1 * RT.a2 + c2 * RT.b2 + c3 * RT.c2,
            c1 * RT.a3 + c2 * RT.b3 + c3 * RT.c3,
            a1 * RT.tx + a2 * RT.ty + a3 * RT.tz + tx,
            b1 * RT.tx + b2 * RT.ty + b3 * RT.tz + ty,
            c1 * RT.tx + c2 * RT.ty + c3 * RT.tz + tz};
    }
    inline Point3_<_Float> rotatePoint(const Point3_<_Float>& pt) const
    {
        _Float x = a1 * pt.x + a2 * pt.y + a3 * pt.z;
        _Float y = b1 * pt.x + b2 * pt.y + b3 * pt.z;
        _Float z = c1 * pt.x + c2 * pt.y + c3 * pt.z;
        return {x, y, z};
    }
    inline void rotatePoints(
        const Point3_<_Float>* src, Point3_<_Float>* dst, size_t num) const
    {
        RotateImpl<_Float>::proc(*this, src, dst, num);
    }
    inline Point3_<_Float> transformPoint(const Point3_<_Float>& pt) const
    {
        _Float x = a1 * pt.x + a2 * pt.y + a3 * pt.z + tx;
        _Float y = b1 * pt.x + b2 * pt.y + b3 * pt.z + ty;
        _Float z = c1 * pt.x + c2 * pt.y + c3 * pt.z + tz;
        return {x, y, z};
    }
    inline void transformPoints(
        const Point3_<_Float>* src, Point3_<_Float>* dst, size_t num) const
    {
        TransformImpl<_Float>::proc(*this, src, dst, num);
    }
    template <typename _Tp, class = typename std::enable_if<
                                std::is_floating_point<_Tp>::value>::type>
    void copyTo(RT_<_Tp>& RT) const
    {
        RT.a1 = (_Tp)a1, RT.a2 = (_Tp)a2, RT.a3 = (_Tp)a3;
        RT.b1 = (_Tp)b1, RT.b2 = (_Tp)b2, RT.b3 = (_Tp)b3;
        RT.c1 = (_Tp)c1, RT.c2 = (_Tp)c2, RT.c3 = (_Tp)c3;
        RT.tx = (_Tp)tx, RT.ty = (_Tp)ty, RT.tz = (_Tp)tz;
    }
    template <typename _Tp,
        class = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
    void copyTo(OutArr_<_Tp> arr) const
    {
        assert(!arr.empty() && arr.width == 4 && arr.height >= 3);
        for (int y = 0; y < 3; ++y) {
            _Tp*          row = arr.ptr(y);
            const _Float* ptr = &a1 + (y << 2);
            for (int x = 0; x < 4; ++x) row[x] = ptr[x];
        }
    }
    template <typename _CVMAT,
        class = typename std::enable_if<is_cvmat<_CVMAT>::value>::type>
    void copyTo(_CVMAT& mat) const
    {
        assert(mat.rows >= 3 && mat.cols == 4);
        if (mat.type() == 6) {
            memcpy(mat.data, &a1, sizeof(double[12]));
        } else {
            memcpy(mat.data, &a1, sizeof(float[12]));
        }
    }
    inline Point3_<_Float> center() const
    {
        return {-(a1 * tx + b1 * ty + c1 * tz), -(a2 * tx + b2 * ty + c2 * tz),
            -(a3 * tx + b3 * ty + c3 * tz)};
    }
    inline RT_& setTranslation(_Float Tx, _Float Ty, _Float Tz)
    {
        tx = Tx, ty = Ty, tz = Tz;
        return *this;
    }
    bool operator==(const RT_<_Float>& RT) const
    {
        return a1 == RT.a1 && a2 == RT.a2 && a3 == RT.a3 && b1 == RT.b1 &&
               b2 == RT.b2 && b3 == RT.b3 && c1 == RT.c1 && c2 == RT.c2 &&
               c3 == RT.c3 && tx == RT.tx && ty == RT.ty && tz == RT.tz;
    }
    bool isIdentity() const { return *this == eye(); }

    _Float a1, a2, a3, tx, b1, b2, b3, ty, c1, c2, c3, tz;
};

template <typename _Float> struct RotateImpl {
    static inline void proc(const RT_<_Float>& RT, const Point3_<_Float>* src,
        Point3_<_Float>* dst, size_t num)
    {
        if (RT.isIdentity() && src == dst) return;
        if (!(src != nullptr && dst != nullptr && num > 0)) return;
        for (size_t i = 0; i < num; ++i) dst[i] = RT.rotatePoint(src[i]);
    }
};

template <typename _Float> struct TransformImpl {
    static inline void proc(const RT_<_Float>& RT, const Point3_<_Float>* src,
        Point3_<_Float>* dst, size_t num)
    {
        if (RT.isIdentity() && src == dst) return;
        if (!(src != nullptr && dst != nullptr && num > 0)) return;
        for (size_t i = 0; i < num; ++i) dst[i] = RT.transformPoint(src[i]);
    }
};

template <bool _MultiThread = defaultMultiThreadFlag> struct ParallelFor {
    static inline void proc(
        int start, int end, std::function<void(int, int)> func)
    {
        if (func) func(start, end);
    }
};
template <> struct ParallelFor<true> {
    static inline void proc(
        int start, int end, std::function<void(int, int)> func)
    {
        if (!func) return;
        int cur, cnt = end - start;
        if (cnt <= 1) func(start, end);
#ifdef RGBTSDF_USE_OMP
        int step = (cnt + omp_get_max_threads() - 1) / omp_get_max_threads();
#pragma omp parallel private(cur)
        {
            cur   = omp_get_thread_num();  // Get Thread ID
            int m = step * cur + start;
            int n = (std::min)(step * (cur + 1), cnt) + start;
            func(m, n);
        }
#else
        int thread_num = std::thread::hardware_concurrency();
        int step       = (cnt + thread_num - 1) / thread_num;
        if (step <= 0) return;
        std::vector<std::future<void>> f(thread_num);
        for (cur = 0; cur < thread_num; ++cur) {
            int m  = step * cur + start;
            int n  = (std::min)(step * (cur + 1), cnt) + start;
            f[cur] = std::async(std::launch::async, func, m, n);
        }
        for (cur = 0; cur < thread_num; ++cur) f[cur].wait();
#endif
    }
};

template <typename _Float> static inline void depthToPoints(
    InArr_<_Float> img, K_<_Float> cam, std::vector<Point3_<_Float>>& points)
{
    assert(!img.empty());
    std::vector<_Float> xvec(img.width);
    for (int x = 0; x < img.width; ++x) xvec[x] = (x - cam.cx) / cam.fx;
    for (int y = 0; y < img.height; ++y) {
        auto*  cur = img.ptr(y);
        _Float y0  = (y - cam.cy) / cam.fy;
        for (int x = 0; x < img.width; ++x, ++cur) {
            if (*cur <= 0) continue;
            points.emplace_back(*cur * xvec[x], *cur * y0, *cur);
        }
    }
}

template <typename _Float> static inline void computeGaussianWeightMap(
    OutArr_<_Float> map, double sigma = 2.0)
{
    static const _Float PI = (_Float)RGBTSDF_PI;
    assert(map.data && map.height > 0 && map.width > 0);
    const _Float        _2sigma2      = (_Float)(2 * sigma * sigma);
    const _Float        sqrt_sigma2pi = (_Float)(sigma * std::sqrt(PI * 2));
    std::vector<_Float> xvec(map.width);
    _Float wh2 = (_Float)(map.width * map.width + map.height * map.height) / 2;
    const _Float half_width  = (_Float)map.width / 2;
    const _Float half_height = (_Float)map.height / 2;
    for (int x = 0; x < map.width; ++x) xvec[x] = x - half_width;
    for (int y = 0; y < map.height; ++y) {
        _Float* cur = map.ptr(y);
        _Float  y0  = y - half_height;
        for (int x = 0; x < map.width; ++x, ++cur) {
            _Float x0 = xvec[x];
            *cur = exp(-(x0 * x0 + y0 * y0) / wh2 / _2sigma2) / sqrt_sigma2pi;
        }
    }
}

// clang-format off
template <TSDFType _Type, typename _Float> struct InitVoxelImpl;
template <TSDFType _Type, typename _Float> struct ExtractVoxelImpl;
template <TSDFType _Type, typename _Float> struct ExtractPointImpl;
template <TSDFType _Type, typename _Float> struct ExtractTriMeshImpl;
template <TSDFType _Type, typename _Float, bool _Inverse>  struct IntegrateImpl;
// clang-format on

template <TSDFType _Type, typename _Float, int _VoxelSize> struct TSDFVolume_ {
    static_assert(_VoxelSize > 0, "_VoxelSize Should Bigger Than Zero.");
    constexpr static int _MAX_DEPTH = MaxBinaryExponent<_VoxelSize>::value;
    using _Index =
        typename std::conditional<_MAX_DEPTH <= 10, int, long long>::type;
    constexpr static int    _GETX        = (1 << _MAX_DEPTH) - 1;
    constexpr static int    _GETY        = _GETX << _MAX_DEPTH;
    constexpr static int    _GETZ        = _GETY << _MAX_DEPTH;
    constexpr static int    _UNITX       = 1;
    constexpr static int    _UNITY       = _UNITX << _MAX_DEPTH;
    constexpr static int    _UNITZ       = _UNITY << _MAX_DEPTH;
    constexpr static int    _SHIFTX      = 0;
    constexpr static int    _SHIFTY      = _SHIFTX + _MAX_DEPTH;
    constexpr static int    _SHIFTZ      = _SHIFTY + _MAX_DEPTH;
    constexpr static int    _INVALID     = ~(_GETX | _GETY | _GETZ);
    constexpr static size_t _VOLUME_AREA = _VoxelSize * _VoxelSize;
    constexpr static size_t _VOLUME_SIZE = _VOLUME_AREA * _VoxelSize;
    TSDFVolume_()                        = default;
    TSDFVolume_(const TSDFVolume_&)      = default;
    // clang-format off
    _Index LEFTVoxel(_Index voxelID) const { return !(voxelID&_GETX)?-1:(voxelID-_UNITX); }
    _Index RIGHTVoxel(_Index voxelID) const { return (voxelID&_GETX)==_GETX?-1:(voxelID+_UNITX); }
    _Index BACKVoxel(_Index voxelID) const { return !(voxelID&_GETY)?-1:(voxelID-_UNITY); }
    _Index FRONTVoxel(_Index voxelID) const { return (voxelID&_GETY)==_GETY?-1:(voxelID+_UNITY); }
    _Index TOPVoxel(_Index voxelID) const { return (voxelID&_GETZ)==_GETZ?-1:(voxelID+_UNITZ); }
    _Index BOTTOMVoxel(_Index voxelID) const { return !(voxelID&_GETZ)?-1:(voxelID-_UNITZ); }
    // clang-format on
    static inline _Index getVoxelID(_Index xi, _Index yi, _Index zi)
    {
        return (xi << _SHIFTX) | (yi << _SHIFTY) | (zi << _SHIFTZ);
    }
    template <bool _Inverse>
    inline void integrateDepthImage(const K_<_Float>& cam,
        const RT_<_Float>& rt, coord_trait_t<_Float> nearZ,
        coord_trait_t<_Float> farZ, const Point3_<_Float>& origin,
        _Float unit_length, _Float trunc_length, const InArr_<_Float>& depth,
        const InArr_<RGBPixel>& color, const _Float* imgToWeights = 0)
    {
        IntegrateImpl<_Type, _Float, _Inverse>::proc(voxelData_, _VoxelSize,
            cam, rt, nearZ, farZ, origin, unit_length, trunc_length, 0, depth,
            color, imgToWeights);
    }
    inline void extractPointCloud(std::vector<Point3_<_Float>>& points,
        std::vector<RGBPixel>& pixels, const Point3_<_Float>& origin,
        _Float unit_length, _Float minWeight)
    {
        ExtractPointImpl<_Type, _Float>::proc(voxelData_, _VoxelSize, points,
            pixels, origin, unit_length, minWeight, nullptr, nullptr, nullptr);
    }
    inline TSDFVoxel_<_Type, _Float>& at(_Index voxelID)
    {
        return const_cast<TSDFVoxel_<_Type, _Float>&>(
            const_cast<const TSDFVolume_*>(this)->at(voxelID));
    }
    inline const TSDFVoxel_<_Type, _Float>& at(_Index voxelID) const
    {
        assert(!(voxelID & _INVALID));
        return at((voxelID & _GETX) >> _SHIFTX, (voxelID & _GETY) >> _SHIFTY,
            (voxelID & _GETZ) >> _SHIFTZ);
    }
    inline TSDFVoxel_<_Type, _Float>& at(_Index xi, _Index yi, _Index zi)
    {
        return const_cast<TSDFVoxel_<_Type, _Float>&>(
            const_cast<const TSDFVolume_*>(this)->at(xi, yi, zi));
    }
    inline const TSDFVoxel_<_Type, _Float>& at(
        _Index xi, _Index yi, _Index zi) const
    {
        assert(xi >= 0 && yi >= 0 && zi >= 0 && xi < _VoxelSize &&
               yi < _VoxelSize && zi < _VoxelSize);
        return voxelData_[zi * _VOLUME_AREA + (size_t)yi * _VoxelSize + xi];
    }
    TSDFVoxel_<_Type, _Float> voxelData_[_VOLUME_SIZE];
};
template <TSDFType _Type, typename _Float>
struct TSDFVolume_<_Type, _Float, -1> {
    using _Index                    = int;
    TSDFVolume_()                   = delete;
    TSDFVolume_(const TSDFVolume_&) = delete;
    TSDFVolume_(_Index voxelSize)
        : maxDepth_(maxBinaryExponent(voxelSize))
        , voxelSize_(voxelSize)
        , getX_((1 << maxDepth_) - 1)
        , getY_(getX_ << maxDepth_)
        , getZ_(getY_ << maxDepth_)
        , unitX_(1)
        , unitY_(unitX_ << maxDepth_)
        , unitZ_(unitY_ << maxDepth_)
        , shiftX_(0)
        , shiftY_(shiftX_ + maxDepth_)
        , shiftZ_(shiftY_ + maxDepth_)
        , invalidID_(~(getX_ | getY_ | getZ_))
        , volumeArea_(voxelSize_ * voxelSize_)
        , volumeSize_(volumeArea_ * voxelSize_)
    {
        assert(voxelSize > 0);
        size_t volumeSize = (size_t)voxelSize * voxelSize * voxelSize;
        voxelData_        = new TSDFVoxel_<_Type, _Float>[volumeSize];
        assert(voxelData_ != nullptr);
        for (size_t i = 0; i < volumeSize; ++i)
            voxelData_[i] = TSDFDefaultValue<_Type, _Float>::value;
    }
    ~TSDFVolume_()
    {
        if (voxelData_) delete[] voxelData_;
    }
    static inline _Index maxBinaryExponent(_Index num)
    {
        _Index n = 0;
        for (; 1 << ++n < num;) {}
        return n;
    }
    // clang-format off
    _Index LEFTVoxel(_Index voxelID) const { return !(voxelID&getX_)?-1:(voxelID-unitX_); }
    _Index RIGHTVoxel(_Index voxelID) const { return (voxelID&getX_)==getX_?-1:(voxelID+unitX_); }
    _Index BACKVoxel(_Index voxelID) const { return !(voxelID&getY_)?-1:(voxelID-unitY_); }
    _Index FRONTVoxel(_Index voxelID) const { return (voxelID&getY_)==getY_?-1:(voxelID+unitY_); }
    _Index TOPVoxel(_Index voxelID) const { return (voxelID&getZ_)==getZ_?-1:(voxelID+unitZ_); }
    _Index BOTTOMVoxel(_Index voxelID) const { return !(voxelID&getZ_)?-1:(voxelID-unitZ_); }
    // clang-format on
    _Index getVoxelID(_Index xi, _Index yi, _Index zi) const
    {
        return (xi << shiftX_) | (yi << shiftY_) | (zi << shiftZ_);
    }
    template <bool _Inverse>
    inline void integrateDepthImage(const K_<_Float>& cam,
        const RT_<_Float>& rt, coord_trait_t<_Float> nearZ,
        coord_trait_t<_Float> farZ, const Point3_<_Float>& origin,
        _Float unit_length, _Float trunc_length, const InArr_<_Float>& depth,
        const InArr_<RGBPixel>& color, const _Float* imgToWeights = nullptr)
    {
        IntegrateImpl<_Type, _Float, _Inverse>::proc(voxelData_, voxelSize_,
            cam, rt, nearZ, farZ, origin, unit_length, trunc_length, 0, depth,
            color, imgToWeights);
    }
    inline void extractPointCloud(std::vector<Point3_<_Float>>& points,
        std::vector<RGBPixel>& pixels, const Point3_<_Float>& origin,
        _Float unit_length, _Float minWeight)
    {
        ExtractPointImpl<_Type, _Float>::proc(voxelData_, voxelSize_, points,
            pixels, origin, unit_length, minWeight, nullptr, nullptr, nullptr);
    }
    inline TSDFVoxel_<_Type, _Float>& at(_Index voxelID)
    {
        return const_cast<TSDFVoxel_<_Type, _Float>&>(
            const_cast<const TSDFVolume_*>(this)->at(voxelID));
    }
    inline const TSDFVoxel_<_Type, _Float>& at(_Index voxelID) const
    {
        assert(!(voxelID & invalidID_));
        return at((voxelID & getX_) >> shiftX_, (voxelID & getY_) >> shiftY_,
            (voxelID & getZ_) >> shiftZ_);
    }
    inline TSDFVoxel_<_Type, _Float>& at(_Index xi, _Index yi, _Index zi)
    {
        assert(xi >= 0 && xi < voxelSize_ && yi >= 0 && yi < voxelSize_ &&
               zi >= 0 && zi < voxelSize_);
        return voxelData_[(zi * voxelSize_ + yi) * voxelSize_ + xi];
    }
    inline const TSDFVoxel_<_Type, _Float>& at(
        _Index xi, _Index yi, _Index zi) const
    {
        assert(xi >= 0 && xi < voxelSize_ && yi >= 0 && yi < voxelSize_ &&
               zi >= 0 && zi < voxelSize_);
        return voxelData_[(zi * voxelSize_ + yi) * voxelSize_ + xi];
    }
    const _Index               maxDepth_;
    const _Index               voxelSize_;
    const _Index               getX_, getY_, getZ_;
    const _Index               unitX_, unitY_, unitZ_;
    const _Index               shiftX_, shiftY_, shiftZ_;
    const _Index               invalidID_;
    const size_t               volumeArea_;
    const size_t               volumeSize_;
    TSDFVoxel_<_Type, _Float>* voxelData_;
};

template <TSDFType _Type, typename _Float, int _Layer>
struct HierachicalTSDFVolume_ {};

template <size_t _ElementSize, int _BlockSize> struct OcTreeAllocator_ {
    static_assert(_ElementSize > 0 && _BlockSize > 0,
        "_ElementSize And _BlockSize Should Bigger Than Zero.");
    constexpr static size_t _BlockMemSize = _ElementSize * _BlockSize;
    OcTreeAllocator_(void* (*creator)(size_t), void (*destroyer)(void*))
        : creator_(creator), destroyer_(destroyer), cur_(nullptr), end_(nullptr)
    {}
    OcTreeAllocator_(const OcTreeAllocator_&) = delete;
    ~OcTreeAllocator_() { free(); }
    void* alloc()
    {
        if (!lost_.empty()) {
            void* ptr = lost_.top();
            assert(ptr);
            lost_.pop();
            return ptr;
        }
        if (cur_ == end_) {
            cur_ = (char(*)[_ElementSize])creator_(_BlockMemSize);
            assert(cur_);
            end_ = cur_ + _BlockSize;
            mem_.push((char*)cur_);
        }
        assert(cur_);
        auto ret = cur_++;
        return ret;
    }
    void free(void* ptr)
    {
        if (ptr) lost_.push((char*)ptr);
    }
    void free()
    {
        cur_ = end_ = nullptr;
        std::stack<char*>().swap(lost_);
        while (!mem_.empty()) destroyer_(mem_.top()), mem_.pop();
    }
    void freeWhenPossible()
    {
        assert(!(mem_.empty() && cur_));
        if (mem_.empty()) return;
        size_t useNum = mem_.size() * _BlockSize - _BlockSize;
        useNum += (cur_ - (char(*)[_ElementSize])mem_.top());
        if (useNum == lost_.size()) free();
    }
    std::stack<char*> mem_;
    std::stack<char*> lost_;
    void* (*creator_)(size_t);
    void (*destroyer_)(void*);
    char (*cur_)[_ElementSize];
    char (*end_)[_ElementSize];
};

template <typename _Tp> using NodeAllocator =
    OcTreeAllocator_<sizeof(_Tp), 512>;
template <typename _Tp> struct DefaultAllocator {
    static inline _Tp* alloc()
    {
        auto* memptr = (_Tp*)allocator.alloc();
        assert(memptr);
        memset(memptr, 0, sizeof(_Tp));
        return memptr;
    }
    static void free() { allocator.free(); }
    static void free(void* ptr) { allocator.free(ptr); }
    static void freeWhenPossible() { allocator.freeWhenPossible(); }
    static NodeAllocator<_Tp> allocator;
};
template <typename _Tp>
NodeAllocator<_Tp> DefaultAllocator<_Tp>::allocator(::malloc, ::free);

template <typename _Tp> using VolumeAllocator =
    OcTreeAllocator_<sizeof(_Tp), 256>;
template <TSDFType _Type, typename _Float, int _VoxelSize>
struct DefaultAllocator<TSDFVolume_<_Type, _Float, _VoxelSize>> {
    using _VolumeData = TSDFVolume_<_Type, _Float, _VoxelSize>;
    static inline _VolumeData* alloc()
    {
        auto* memptr = (_VolumeData*)allocator.alloc();
        assert(memptr);
        InitVoxelImpl<_Type, _Float>::proc(
            memptr->voxelData_, _VolumeData::_VOLUME_SIZE);
        return memptr;
    }
    static void free() { allocator.free(); }
    static void free(_VolumeData* ptr) { allocator.free(ptr); }
    static void freeWhenPossible() { allocator.freeWhenPossible(); }
    static VolumeAllocator<_VolumeData> allocator;
};
template <TSDFType _Type, typename _Float, int _VoxelSize>
VolumeAllocator<TSDFVolume_<_Type, _Float, _VoxelSize>>
    DefaultAllocator<TSDFVolume_<_Type, _Float, _VoxelSize>>::allocator(
        ::malloc, ::free);

template <int _Depth, int _MAX_DEPTH, typename _VolumeData, typename _Index>
struct OctNode_ {
    static_assert(_Depth < _MAX_DEPTH, "_Depth Should Less Than _MaxDepth.");
    static_assert(_MAX_DEPTH <= sizeof(_Index) * 0x8 / 0x3,
        "_MaxDepth Not Match _Index Type, Consider Using Int64.");
    using _NodeData = OctNode_<_Depth - 0x1, _MAX_DEPTH, _VolumeData, _Index>;
    static_assert(sizeof(void* [0x8]) == sizeof(_NodeData),
        "Check The Size Of Node, It Means You Should Change The SubNode Size.");
    constexpr static _Index MASKX   = 0x1 << _Depth;
    constexpr static _Index MASKY   = MASKX << _MAX_DEPTH;
    constexpr static _Index MASKZ   = MASKY << _MAX_DEPTH;
    constexpr static _Index SHIFTX  = _Depth;
    constexpr static _Index SHIFTY  = SHIFTX + _MAX_DEPTH - 0x1;
    constexpr static _Index SHIFTZ  = SHIFTY + _MAX_DEPTH - 0x1;
    constexpr static _Index OFFSETX = SHIFTX;
    constexpr static _Index OFFSETY = SHIFTY - 0x1;
    constexpr static _Index OFFSETZ = SHIFTZ - 0x2;
    OctNode_() : nodes {} {}
    _VolumeData* openVolume(_Index nodeID)
    {
        _Index c = (nodeID & MASKX) >> SHIFTX;
        c |= (nodeID & MASKY) >> SHIFTY;
        c |= (nodeID & MASKZ) >> SHIFTZ;
        if (!nodes[c])
            nodes[c] = (_NodeData*)DefaultAllocator<void* [0x8]>::alloc();
        return nodes[c]->openVolume(nodeID);
    }
    _VolumeData* findVolume(_Index nodeID)
    {
        const _VolumeData* ptr =
            const_cast<const OctNode_*>(this)->findVolume(nodeID);
        return const_cast<_VolumeData*>(ptr);
    }
    const _VolumeData* findVolume(_Index nodeID) const
    {
        _Index c = (nodeID & MASKX) >> SHIFTX;
        c |= (nodeID & MASKY) >> SHIFTY;
        c |= (nodeID & MASKZ) >> SHIFTZ;
        if (!nodes[c]) return nullptr;
        return nodes[c]->findVolume(nodeID);
    }
    void traverseData(std::function<void(_Index, const _VolumeData*)> func,
        _Index nodeID = 0) const
    {
        for (_Index i = 0; i < 0x8; ++i) {
            _Index c = nodeID;
            c |= (i & 0x1) << SHIFTX;
            c |= (i & 0x2) << SHIFTY;
            c |= (i & 0x4) << SHIFTZ;
            if (!nodes[i]) continue;
            nodes[i]->traverseData(func, c);
        }
    }
    void freeData()
    {
        for (_Index i = 0; i < 0x8; ++i) {
            if (nodes[i]) {
                nodes[i]->freeData();
                DefaultAllocator<void* [0x8]>::free(nodes[i]);
                nodes[i] = nullptr;
            }
        }
    }
    void freeData(_Index nodeID)
    {
        _Index c = ((nodeID & MASKX) >> OFFSETX) +
                   ((nodeID & MASKY) >> OFFSETY) +
                   ((nodeID & MASKZ) >> OFFSETZ);
        if (!nodes[c]) {
            nodes[c]->freeData(nodeID);
            bool flag = true;
            for (_Index i = 0; i < 0x8; ++i) {
                if (nodes[c][i]) flag = false;
            }
            if (flag) {
                DefaultAllocator<void* [0x8]>::free(nodes[c]);
                nodes[c] = nullptr;
            }
        }
    }
    _NodeData* nodes[0x8];  // Top-Bottom Front-Back Left-Right
};
template <int _MAX_DEPTH, typename _VolumeData, typename _Index>
struct OctNode_<0, _MAX_DEPTH, _VolumeData, _Index> {
    constexpr static _Index MASKX   = 0x1;
    constexpr static _Index MASKY   = MASKX << _MAX_DEPTH;
    constexpr static _Index MASKZ   = MASKY << _MAX_DEPTH;
    constexpr static _Index SHIFTX  = 0x0;
    constexpr static _Index SHIFTY  = SHIFTX + _MAX_DEPTH - 0x1;
    constexpr static _Index SHIFTZ  = SHIFTY + _MAX_DEPTH - 0x1;
    constexpr static _Index OFFSETX = SHIFTX;
    constexpr static _Index OFFSETY = SHIFTY - 0x1;
    constexpr static _Index OFFSETZ = SHIFTZ - 0x2;
    OctNode_()                      = delete;
    OctNode_(const OctNode_&)       = delete;
    _VolumeData* openVolume(_Index nodeID)
    {
        _Index c = nodeID & MASKX;
        c |= (nodeID & MASKY) >> SHIFTY;
        c |= (nodeID & MASKZ) >> SHIFTZ;
        if (!nodes[c]) nodes[c] = DefaultAllocator<_VolumeData>::alloc();
        return nodes[c];
    }
    _VolumeData* findVolume(_Index nodeID)
    {
        const _VolumeData* ptr =
            const_cast<const OctNode_*>(this)->findVolume(nodeID);
        return const_cast<_VolumeData*>(ptr);
    }
    const _VolumeData* findVolume(_Index nodeID) const
    {
        _Index c = nodeID & MASKX;
        c |= (nodeID & MASKY) >> SHIFTY;
        c |= (nodeID & MASKZ) >> SHIFTZ;
        if (!nodes[c]) return nullptr;
        return nodes[c];
    }
    void traverseData(std::function<void(_Index, const _VolumeData*)> func,
        _Index nodeID = 0) const
    {
        for (_Index i = 0; i < 0x8; ++i) {
            _Index c = nodeID;
            c |= (i & 0x1) << SHIFTX;
            c |= (i & 0x2) << SHIFTY;
            c |= (i & 0x4) << SHIFTZ;
            if (!nodes[i]) continue;
            func(c, nodes[i]);
        }
    }
    void freeData()
    {
        for (_Index i = 0; i < 0x8; ++i) {
            DefaultAllocator<_VolumeData>::free(nodes[i]);
            nodes[i] = nullptr;
        }
    }
    void freeData(_Index nodeID)
    {
        _Index c = ((nodeID & MASKX) >> OFFSETX) +
                   ((nodeID & MASKY) >> OFFSETY) +
                   ((nodeID & MASKZ) >> OFFSETZ);
        if (nodes[c]) {
            DefaultAllocator<_VolumeData>::free(nodes[c]);
            nodes[c] = nullptr;
        }
    }
    _VolumeData* nodes[0x8];
};

/** Constraints Struct For Optimize. */
template <typename _Float> struct TSDFConstraint_ {
    unsigned char   axis;  // [0,1,2]
    _Float          lambda;
    _Float          val[2];
    Point3_<_Float> rgb[2];
    Point3_<_Float> point;

    static void extractPoint(const TSDFConstraint_<_Float>& constraint,
        coord_trait_t<_Float> unitLength, Point3_<_Float>& point)
    {
        point = constraint.point;
        (&point.x)[constraint.axis] += constraint.lambda * unitLength;
    }
    static void extractPointAndPixel(const TSDFConstraint_<_Float>& constraint,
        coord_trait_t<_Float> unitLength, Point3_<_Float>& point,
        Point3_<_Float>& rgb)
    {
        point = constraint.point;
        (&point.x)[constraint.axis] += constraint.lambda * unitLength;
        rgb = constraint.rgb[0] +
              constraint.lambda * (constraint.rgb[1] - constraint.rgb[0]);
    }
    static void extractPointAndPixel(const TSDFConstraint_<_Float>& constraint,
        coord_trait_t<_Float> unitLength, Point3_<_Float>& point, RGBPixel& rgb)
    {
        point = constraint.point;
        (&point.x)[constraint.axis] += constraint.lambda * unitLength;
        Point3_<float> color =
            constraint.rgb[0] +
            constraint.lambda * (constraint.rgb[1] - constraint.rgb[0]);
        rgb = {clamp<float>(color.x * 0xFF, 0x0, 0xFF),
            clamp<float>(color.y * 0xFF, 0x0, 0xFF),
            clamp<float>(color.z * 0xFF, 0x0, 0xFF)};
    }
};

template <int _MAX_DEPTH = 0xA, int _VoxelSize = 0x20,
    TSDFType _Type = TSDFType::TSDF, typename _Float = float,
    class _Index =
        typename std::conditional<_MAX_DEPTH <= 0xA, int, long long>::type>
struct OcTree_ {
    static_assert(std::is_floating_point<_Float>::value,
        "_Float Should Be FLOAT OR DOUBLE TYPE");
    static_assert(_VoxelSize > 0, "_VoxelSize Should Bigger Than Zero.");
    static_assert(_MAX_DEPTH <= sizeof(long long) * 0x8 / 0x3,
        "_MaxDepth Is Bigger Than Limits, Consider Using Number 21.");
    // constexpr static int _VoxelSize   = _VoxelSize;
    // constexpr static int _VoxelSquare = _VoxelSize * _VoxelSize;
    using Index                      = _Index;
    using _Point                     = Point3_<_Float>;
    using _VoxelData                 = TSDFVoxel_<_Type, _Float>;
    using _VolumeData                = TSDFVolume_<_Type, _Float, _VoxelSize>;
    using _MapPointData              = std::unordered_map<long long,
                     std::tuple<std::vector<_Point>, std::vector<RGBPixel>>>;
    constexpr static _Index _GETX    = (0x1 << _MAX_DEPTH) - 0x1;
    constexpr static _Index _GETY    = _GETX << _MAX_DEPTH;
    constexpr static _Index _GETZ    = _GETY << _MAX_DEPTH;
    constexpr static _Index _UNITX   = 0x1;
    constexpr static _Index _UNITY   = _UNITX << _MAX_DEPTH;
    constexpr static _Index _UNITZ   = _UNITY << _MAX_DEPTH;
    constexpr static _Index _SHIFTX  = 0x0;
    constexpr static _Index _SHIFTY  = _SHIFTX + _MAX_DEPTH;
    constexpr static _Index _SHIFTZ  = _SHIFTY + _MAX_DEPTH;
    constexpr static _Index _INVALID = ~(_GETX | _GETY | _GETZ);
    OcTree_()                        = delete;
    OcTree_(const OcTree_&)          = delete;
    OcTree_(double unit_length, const Point3_<double>& center = {})
        : cur_time_(0)
        , unit_length_(unit_length)
        , half_length_(unit_length / 0x2)
        , trunc_length_(unit_length * RGBTSDF_TRUNCATED_FACTOR)
        , volume_length_(unit_length * _VoxelSize)
        , octree_length_(volume_length_ * (0x1LL << _MAX_DEPTH))
        , octree_origin_(center - octree_length_ / 0x2)
    {}
    ~OcTree_() { clear(); }
    void clear()
    {
        std::lock_guard<std::mutex> lock {lock_};
        cur_time_ = 0;
        root_node_.freeData();
        DefaultAllocator<void* [0x8]>::freeWhenPossible();
        DefaultAllocator<_VolumeData>::freeWhenPossible();
    }
    double getTruncatedDistance() const noexcept { return trunc_length_; }
    void   setTruncatedDistance(double truncDist) noexcept
    {
        trunc_length_ = truncDist;
    }
    Point3_<double> centerPoint() const
    {
        return octree_origin_ + octree_length_ / 0x2;
    }
    // clang-format off
    _Index LEFTNode(_Index volumeID) const { return !(volumeID&_GETX) ? -1 : (volumeID-_UNITX); }
    _Index RIGHTNode(_Index volumeID) const { return (volumeID&_GETX)==_GETX ? -1 : (volumeID+_UNITX); }
    _Index BACKNode(_Index volumeID) const { return !(volumeID&_GETY) ? -1 : (volumeID-_UNITY); }
    _Index FRONTNode(_Index volumeID) const { return (volumeID&_GETY)==_GETY ? -1 : (volumeID+_UNITY); }
    _Index TOPNode(_Index volumeID) const { return (volumeID&_GETZ)==_GETZ ? -1 : (volumeID+_UNITZ); }
    _Index BOTTOMNode(_Index volumeID) const { return !(volumeID&_GETZ) ? -1 : (volumeID-_UNITZ); }
    // clang-format on
    _VolumeData* openVolume(_Index volumeID)
    {
        std::lock_guard<std::mutex> lock {lock_};
        if (volumeID & _INVALID) return nullptr;
        return root_node_.openVolume(volumeID);
    }
    _VolumeData* findVolume(_Index volumeID)
    {
        if (volumeID & _INVALID) return nullptr;
        return root_node_.findVolume(volumeID);
    }
    const _VolumeData* findVolume(_Index volumeID) const
    {
        if (volumeID & _INVALID) return nullptr;
        return root_node_.findVolume(volumeID);
    }
    template <typename _Tp = _Float>
    _Index pointToIndex(const Point3_<_Tp>& pt) const
    {
        Point3_<double> offset = {pt.x - octree_origin_.x,
            pt.y - octree_origin_.y, pt.z - octree_origin_.z};
        if (offset >= 0 && offset < octree_length_) {
            _Index c = 0x0;
            c |= (_Index)(offset.x / volume_length_) << _SHIFTX;
            c |= (_Index)(offset.y / volume_length_) << _SHIFTY;
            c |= (_Index)(offset.z / volume_length_) << _SHIFTZ;
            return c;
        }
        return -1;
    }
    template <typename _Tp = _Float>
    Point3_<_Tp> indexToPoint(_Index volumeID) const
    {
        constexpr static _Float none = std::numeric_limits<_Float>::quiet_NaN();
        if (volumeID & _INVALID) return {none, none, none};
        Point3_<double> pt = octree_origin_;
        pt.x += (double)((volumeID & _GETX) >> _SHIFTX) * volume_length_;
        pt.y += (double)((volumeID & _GETY) >> _SHIFTY) * volume_length_;
        pt.z += (double)((volumeID & _GETZ) >> _SHIFTZ) * volume_length_;
        return {pt.x, pt.y, pt.z};
    }
    _Index getIndexByOffset(
        _Index volumeID, _Index offsetX, _Index offsetY, _Index offsetZ) const
    {
        _Index c = 0;
        c |= (volumeID & _GETX) + (offsetX << _SHIFTX);
        c |= (volumeID & _GETY) + (offsetY << _SHIFTY);
        c |= (volumeID & _GETZ) + (offsetZ << _SHIFTZ);
        return c;
    }
    Point3_<int> getIndexOffset(_Index oldID, _Index newID) const
    {
        int xi = ((newID & _GETX) - (oldID & _GETX)) >> _SHIFTX;
        int yi = ((newID & _GETY) - (oldID & _GETY)) >> _SHIFTY;
        int zi = ((newID & _GETZ) - (oldID & _GETZ)) >> _SHIFTZ;
        return {xi, yi, zi};
    }
    const _VoxelData* findVoxel(const _Point& point) const
    {
        _Index volumeID = pointToIndex(point);
        if (volumeID & _INVALID) return nullptr;
        auto volumePtr = root_node_.findVolume(volumeID);
        if (!volumePtr) return nullptr;
        Point3_<double> voxelPt = {point.x - octree_origin_.x,
            point.y - octree_origin_.y, point.z - octree_origin_.z};
        voxelPt.x -= (double)((volumeID & _GETX) >> _SHIFTX) * volume_length_;
        voxelPt.y -= (double)((volumeID & _GETY) >> _SHIFTY) * volume_length_;
        voxelPt.z -= (double)((volumeID & _GETZ) >> _SHIFTZ) * volume_length_;
        voxelPt /= unit_length_;
        return &(volumePtr->at((typename _VolumeData::_Index)voxelPt.x,
            (typename _VolumeData::_Index)voxelPt.y,
            (typename _VolumeData::_Index)voxelPt.z));
    }
    std::vector<_Index> depthToIndexs(InArr_<_Float> img, const K_<_Float>& cam,
        const RT_<_Float>& rt, _Float nearZ, _Float farZ) const
    {
        assert(!img.empty());
        const _Index truncatedFactor =
            (_Index)std::ceil(trunc_length_ / unit_length_);
        RT_<_Float>         rt_inv = rt.inv();
        std::vector<_Float> xvec(img.width);
        for (int x = 0; x < img.width; ++x)
            xvec[x] = ((_Float)x - cam.cx) / cam.fx;
        // calculate total number first.
        int validCnt = img.countNonZero();
#ifndef RGBTSDF_USE_3X3X3_NEIGHBOR
        std::vector<_Index> inds(validCnt * 0x7LL);
#else
        std::vector<_Index> inds(validCnt * 0x1BLL);
#endif
        validCnt = 0;  // reset validCnt to Zero.
        for (int y = 0; y < img.height; ++y) {
            auto*  cur = img.ptr(y);
            _Float y0  = ((_Float)y - cam.cy) / cam.fy;
            for (int x = 0; x < img.width; ++x, ++cur) {
                if (!(*cur > nearZ && *cur < farZ)) continue;
                _Point pt = {*cur * xvec[x], *cur * y0, *cur};
                _Float xc = rt_inv.a1 * pt.x + rt_inv.a2 * pt.y +
                            rt_inv.a3 * pt.z + rt_inv.tx;
                _Float yc = rt_inv.b1 * pt.x + rt_inv.b2 * pt.y +
                            rt_inv.b3 * pt.z + rt_inv.ty;
                _Float zc = rt_inv.c1 * pt.x + rt_inv.c2 * pt.y +
                            rt_inv.c3 * pt.z + rt_inv.tz;
                _Index id = pointToIndex({xc, yc, zc});
                _Point po = indexToPoint(id);
                _Index xi = (_Index)((xc - po.x) / (_Float)unit_length_);
                _Index yi = (_Index)((yc - po.y) / (_Float)unit_length_);
                _Index zi = (_Index)((zc - po.z) / (_Float)unit_length_);
                // increment record index
                inds[validCnt++] = id;
                // clang-format off
#ifndef RGBTSDF_USE_3X3X3_NEIGHBOR
                if (zi < truncatedFactor) inds[validCnt++] = BOTTOMNode(id);
                if (yi < truncatedFactor) inds[validCnt++] = BACKNode(id);
                if (xi < truncatedFactor) inds[validCnt++] = LEFTNode(id);
                if (zi + truncatedFactor >= _VoxelSize) inds[validCnt++] = TOPNode(id);
                if (yi + truncatedFactor >= _VoxelSize) inds[validCnt++] = FRONTNode(id);
                if (xi + truncatedFactor >= _VoxelSize) inds[validCnt++] = RIGHTNode(id);
#else
                bool left = xi < truncatedFactor;
                bool back = yi < truncatedFactor;
                bool bottom = zi < truncatedFactor;
                bool top = zi + truncatedFactor >= _VoxelSize;
                bool front = yi + truncatedFactor >= _VoxelSize;
                bool right = xi + truncatedFactor >= _VoxelSize;
                if (top) inds[validCnt++] = TOPNode(id);
                if (back) inds[validCnt++] = BACKNode(id);
                if (left) inds[validCnt++] = LEFTNode(id);
                if (right) inds[validCnt++] = RIGHTNode(id);
                if (front) inds[validCnt++] = FRONTNode(id);
                if (bottom) inds[validCnt++] = BOTTOMNode(id);
                if (left && back) inds[validCnt++] = LEFTNode(BACKNode(id));
                if (left && bottom) inds[validCnt++] = LEFTNode(BOTTOMNode(id));
                if (left && top) inds[validCnt++] = LEFTNode(TOPNode(id));
                if (left && front) inds[validCnt++] = LEFTNode(FRONTNode(id));
                if (right && back) inds[validCnt++] = RIGHTNode(BACKNode(id));
                if (right && bottom) inds[validCnt++] = RIGHTNode(BOTTOMNode(id));
                if (right && top) inds[validCnt++] = RIGHTNode(TOPNode(id));
                if (right && front) inds[validCnt++] = RIGHTNode(FRONTNode(id));
                if (back && top) inds[validCnt++] = BACKNode(TOPNode(id));
                if (back && bottom) inds[validCnt++] = BACKNode(BOTTOMNode(id));
                if (front && top) inds[validCnt++] = FRONTNode(TOPNode(id));
                if (front && bottom) inds[validCnt++] = FRONTNode(BOTTOMNode(id));
                if (left && back && top) inds[validCnt++] = LEFTNode(BACKNode(TOPNode(id)));
                if (left && back && bottom) inds[validCnt++] = LEFTNode(BACKNode(BOTTOMNode(id)));
                if (left && front && top) inds[validCnt++] = LEFTNode(FRONTNode(TOPNode(id)));
                if (left && front && bottom) inds[validCnt++] = LEFTNode(FRONTNode(BOTTOMNode(id)));
                if (right && back && top) inds[validCnt++] = RIGHTNode(BACKNode(TOPNode(id)));
                if (right && back && bottom) inds[validCnt++] = RIGHTNode(BACKNode(BOTTOMNode(id)));
                if (right && front && top) inds[validCnt++] = RIGHTNode(FRONTNode(TOPNode(id)));
                if (right && front && bottom) inds[validCnt++] = RIGHTNode(FRONTNode(BOTTOMNode(id)));
#endif
                // clang-format on
            }
        }
        assert(validCnt <= (int)inds.size());
        std::set<_Index> indset(inds.begin(), inds.begin() + validCnt);
        inds.assign(indset.begin(), indset.end());
        return inds;
    }
    std::vector<_Index> getBoundingBoxIndexs(
        Point3_<_Float> pt, Point3_<_Float> len) const
    {
        std::vector<_Index> inds;
        const _Index        id = pointToIndex(pt);
        for (int z = 0; z < len.z / volume_length_; ++z) {
            _Index idy = id;
            for (int y = 0; y < len.y / volume_length_; ++y) {
                _Index idx = idy;
                for (int x = 0; x < len.x / volume_length_; ++x) {
                    inds.emplace_back(
                        id + z * _UNITZ + y * _UNITY + x * _UNITX);
                }
            }
        }
        return inds;
    }
    template <bool _MultiThread = defaultMultiThreadFlag> void rasterDepthImage(
        const K_<_Float>& cam, const RT_<_Float>& rt, _Float nearZ, _Float farZ,
        OutArr_<_Float> depth, OutArr_<RGBPixel> color = noArr()) const
    {
        assert(!depth.empty());
        depth.setZero();
        const bool useColor = !color.empty() && color.size() == depth.size();
        if (useColor) color.setZero();
        RT_<_Float>         invRT = rt.inv();
        std::vector<_Float> uvec(depth.width);
        for (int x = 0; x < depth.width; ++x)
            uvec[x] = ((_Float)x - cam.cx) / cam.fx;
        ParallelFor<_MultiThread>::proc(
            0, depth.height, [&, this](int m, int n) {
                for (int y = m; y < n; ++y) {
                    _Float*   dptr   = depth.ptr(y);
                    RGBPixel* rgbptr = useColor ? color.ptr(y) : nullptr;
                    _Float    v0     = ((_Float)y - cam.cy) / cam.fy;
                    for (int x = 0; x < depth.width; ++x, ++rgbptr) {
                        _Point pt = invRT.transformPoint(
                            {uvec[x] * nearZ, v0 * nearZ, nearZ});
                        _Point ptz = invRT.rotatePoint({uvec[x] * unit_length_,
                            v0 * unit_length_, unit_length_});
                        auto*  voxelPtr = findVoxel(pt);
                        _Float curVal   = voxelPtr ? voxelPtr->value() : 1;
                        for (_Float z = nearZ; z < farZ;
                             z += (_Float)unit_length_) {
                            _Float prevVal = curVal;
                            _Point prevPt  = pt;
                            pt += ptz;
                            voxelPtr = findVoxel(pt);
                            curVal   = voxelPtr ? voxelPtr->value() : 1;
                            if (!(prevVal >= 0 && curVal < 0)) continue;
                            _Float tsfd_cur  = getTSDFValue(pt);
                            _Float tsfd_prev = getTSDFValue(prevPt);
                            _Float tsdf_offset =
                                tsfd_prev / (tsfd_cur - tsfd_prev);
                            _Float d = z;
                            if (!((std::abs)(tsdf_offset) < 1)) {
                                d += prevVal * (_Float)unit_length_ /
                                     (curVal - prevVal);
                            } else {
                                d += tsdf_offset * (_Float)unit_length_;
                            }
                            if (!dptr[x] || dptr[x] > d) {
                                dptr[x] = d;
                                if (useColor)
                                    TSDFPixelGetter<_Type, _Float>::get(
                                        voxelPtr, rgbptr);
                                break;
                            }
                        }
                    }
                }
            });
    }
    template <bool _Inverse = false, bool _MultiThread = defaultMultiThreadFlag>
    std::vector<_Index> integrateDepthImage(const K_<_Float>& cam,
        const RT_<_Float>& rt, _Float nearZ, _Float farZ, InArr_<_Float> depth,
        InArr_<RGBPixel> color, InArr_<_Float> i2w = noArr())
    {
        cur_time_++;
        assert(!depth.empty() && (i2w.width == depth.width || !i2w.width) &&
               (i2w.height == depth.height || !i2w.height));
        std::vector<_Index> inds = depthToIndexs(depth, cam, rt, nearZ, farZ);
        ParallelFor<_MultiThread>::proc(
            0, (int)inds.size(), [&, this](int m, int n) {
                for (int i = m; i < n; ++i) {
                    _VolumeData* voxelptr = openVolume(inds[i]);
                    if (!voxelptr) continue;
                    IntegrateImpl<_Type, _Float, _Inverse>::proc(
                        voxelptr->voxelData_, _VoxelSize, cam, &rt.a1, nearZ,
                        farZ, indexToPoint(inds[i]), (_Float)unit_length_,
                        (_Float)trunc_length_, cur_time_, depth, color,
                        {i2w.data, depth.width, depth.height});
                }
            });
        return inds;
    }
    // template <bool _Inverse = false, bool _MultiThread =
    // defaultMultiThreadFlag,
    //     typename... Args>
    // void deIntegrateDepthImage(Args... args)
    // {
    //     integrateDepthImage<!_Inverse, _MultiThread>(
    //         std::forward<Args>(args)...);
    // }
    template <bool _Inverse = false, bool _MultiThread = defaultMultiThreadFlag>
    void integrateDepthAndUpdatePoint(const K_<_Float>& cam,
        const RT_<_Float>& rt, _Float nearZ, _Float farZ, InArr_<_Float> depth,
        InArr_<RGBPixel> color, InArr_<_Float> i2w = noArr(),
        std::function<void(
            long long, std::vector<_Point>&, std::vector<RGBPixel>&)>
               func      = nullptr,
        _Float minWeight = 0)
    {
        auto inds = integrateDepthImage<_Inverse, _MultiThread>(
            cam, rt, nearZ, farZ, depth, color, i2w);
        if (!func) return;
        ParallelFor<_MultiThread>::proc(
            0, (int)inds.size(), [&, this](int m, int n) {
                std::vector<RGBPixel> pixels;
                std::vector<_Point>   points, normals;
                for (int i = m; i < n; ++i) {
                    points.clear(), pixels.clear();
                    extractVolumePoints(inds[i], points, pixels, minWeight);
                    {
                        std::lock_guard<std::mutex> lock {lock_};
                        func(inds[i], points, pixels);
                    }
                }
            });
    }
    template <bool _Inverse = false, bool _MultiThread = defaultMultiThreadFlag>
    void integrateDepthAndUpdatePoint(_MapPointData& mapPoints,
        const K_<_Float>& cam, const RT_<_Float>& rt, _Float nearZ, _Float farZ,
        InArr_<_Float> depth, InArr_<RGBPixel> color,
        InArr_<_Float> i2w = noArr(), _Float minWeight = 0)
    {
        integrateDepthAndUpdatePoint<_Inverse, _MultiThread>(
            cam, rt, nearZ, farZ, depth, color, i2w,
            [&mapPoints](long long volumeID, std::vector<_Point>& points,
                std::vector<RGBPixel>& pixels) {
                auto& result = mapPoints[volumeID];
                std::get<1>(result).swap(pixels);
                std::get<0>(result).swap(points);
            },
            minWeight);
    }
    template <bool _Inverse = false, bool _MultiThread = defaultMultiThreadFlag>
    void integrateDepthAndUpdateTriMesh(const K_<_Float>& cam,
        const RT_<_Float>& rt, _Float nearZ, _Float farZ, InArr_<_Float> depth,
        InArr_<RGBPixel> color, InArr_<_Float> i2w = noArr(),
        std::function<void(long long, std::vector<_Point>&,
            std::vector<RGBPixel>&, std::vector<Point3_<int>>&)>
               func      = nullptr,
        _Float minWeight = 0)
    {
        auto inds = integrateDepthImage<_Inverse, _MultiThread>(
            cam, rt, nearZ, farZ, depth, color, i2w);
        if (!func) return;
        ParallelFor<_MultiThread>::proc(
            0, (int)inds.size(), [&, this](int m, int n) {
                std::vector<Point3_<int>> triIndexs;
                std::vector<RGBPixel>     triPixels;
                std::vector<_Point>       triPoints;
                for (int i = m; i < n; ++i) {
                    triPoints.clear(), triPixels.clear(), triIndexs.clear();
                    extractVolumeTriMeshs(
                        inds[i], triPoints, triPixels, triIndexs, minWeight);
                    {
                        std::lock_guard<std::mutex> lock {lock_};
                        func(inds[i], triPoints, triPixels, triIndexs);
                    }
                }
            });
    }
    template <int _InVoxelSize> inline void integrateVolume(
        const TSDFVolume_<_Type, _Float, _InVoxelSize>& volume, _Index originID)
    {
        static_assert(
            _InVoxelSize % _VoxelSize == 0, "Input Voxel Size Error.");
        constexpr static int    _InSize  = _InVoxelSize / _VoxelSize;
        constexpr static size_t _StrideY = _InVoxelSize;
        constexpr static size_t _StrideZ = _StrideY * _StrideY;
        for (int z = 0; z < _InSize; ++z) {
            for (int y = 0; y < _InSize; ++y)
                for (int x = 0; x < _InSize; ++x) {
                    _Index id = originID + x * _UNITX + y * _UNITY + z * _UNITZ;
                    integrateVolume(id,
                        volume.voxelData_ + z * _VoxelSize * _StrideZ +
                            y * _VoxelSize * _StrideY + x * _VoxelSize,
                        _StrideY, _StrideZ);
                }
        }
    }
    void integrateVolume(const _VolumeData& volumeData, _Index volumeID)
    {
        constexpr static int _VOLUME_SIZE = _VolumeData::_VOLUME_SIZE;
        _VolumeData*         data         = findVolume(volumeID);
        if (!data) {
            data = openVolume(volumeID);
            memcpy(data->voxelData_, volumeData.voxelData_, _VOLUME_SIZE);
        } else {
            for (size_t i = 0; i < _VOLUME_SIZE; ++i)
                data->voxelData_[i] += volumeData.voxelData_[i];
        }
    }
    void integrateVolume(
        _Index volumeID, const _VoxelData* voxelData, int strideY, int strideZ)
    {
        assert(voxelData && strideY >= _VoxelSize && strideZ >= _VoxelSize);
        _VolumeData* data = findVolume(volumeID);
        if (!data) {
            data = openVolume(volumeID);
            if (!data) return;
            _VoxelData*       dataPtr   = data->voxelData_;
            const _VoxelData* voxelZPtr = voxelData;
            for (int z = 0; z < _VoxelSize; ++z, voxelZPtr += strideZ) {
                const _VoxelData* voxelYPtr = voxelZPtr;
                for (int y = 0; y < _VoxelSize;
                     ++y, dataPtr += _VoxelSize, voxelYPtr += strideY)
                    memcpy(dataPtr, voxelYPtr, _VoxelSize * sizeof(_VoxelData));
            }
        } else {
            _VoxelData*       dataPtr   = data->voxelData_;
            const _VoxelData* voxelZPtr = voxelData;
            for (int z = 0; z < _VoxelSize; ++z, voxelZPtr += strideZ) {
                const _VoxelData* voxelYPtr = voxelZPtr;
                for (int y = 0; y < _VoxelSize;
                     ++y, dataPtr += _VoxelSize, voxelYPtr += strideY)
                    for (int x = 0; x < _VoxelSize; ++x)
                        dataPtr[x] += voxelYPtr[x];
            }
        }
    }
    void extractVoxelPoints(std::vector<_Point>& points,
        std::vector<RGBPixel>& pixels, _Float minWeight = 0) const
    {
        int cnt = 0;
        root_node_.traverseData([this, &points, &pixels, &cnt, minWeight](
                                    _Index volumeID, const _VolumeData* data) {
            assert(findVolume(volumeID) == data);
            ExtractVoxelImpl<_Type, _Float>::proc(data->voxelData_, _VoxelSize,
                points, pixels, indexToPoint(volumeID), (_Float)unit_length_,
                minWeight);
            cnt++;
        });
        RGBTSDF_PRINT("extractVoxelPoints - Total Number of Volume is %d", cnt);
    }
    void extractVolumePoints(_Index volumeID, std::vector<_Point>& points,
        std::vector<RGBPixel>& pixels, _Float minWeight = 0) const
    {
        const _VolumeData* cur = findVolume(volumeID);
        if (!cur) return;
        const _VolumeData* top   = findVolume(TOPNode(volumeID));
        const _VolumeData* front = findVolume(FRONTNode(volumeID));
        const _VolumeData* right = findVolume(RIGHTNode(volumeID));
        ExtractPointImpl<_Type, _Float>::proc(cur->voxelData_, _VoxelSize,
            points, pixels, indexToPoint(volumeID), (_Float)unit_length_,
            minWeight, right ? right->voxelData_ : 0,
            front ? front->voxelData_ : 0, top ? top->voxelData_ : 0);
    }
    void extractPointCloud(std::vector<_Point>& points,
        std::vector<RGBPixel>& pixels, _Float minWeight = 0) const
    {
        int cnt = 0;
        root_node_.traverseData([this, &points, &pixels, &cnt, minWeight](
                                    _Index volumeID, const _VolumeData* data) {
            assert(findVolume(volumeID) == data);
            const _VolumeData* top   = findVolume(TOPNode(volumeID));
            const _VolumeData* front = findVolume(FRONTNode(volumeID));
            const _VolumeData* right = findVolume(RIGHTNode(volumeID));
            ExtractPointImpl<_Type, _Float>::proc(data->voxelData_, _VoxelSize,
                points, pixels, indexToPoint(volumeID), (_Float)unit_length_,
                minWeight, right ? right->voxelData_ : 0,
                front ? front->voxelData_ : 0, top ? top->voxelData_ : 0);
            cnt++;
        });
        RGBTSDF_PRINT("extractPointCloud - Total Number of Volume is %d", cnt);
    }
    template <bool _MultiThread = defaultMultiThreadFlag> void extractNormals(
        const _Point* points, _Point* normals, int pointNum) const
    {
        if (!(points && normals && pointNum > 0)) return;
        ParallelFor<_MultiThread>::proc(0, pointNum, [&, this](int m, int n) {
            for (int i = m; i < n; ++i) normals[i] = getTSDFNormal(points[i]);
        });
    }
    template <bool _MultiThread = defaultMultiThreadFlag> void extractNormals(
        const std::vector<_Point>& points, std::vector<_Point>& normals) const
    {
        normals.resize(points.size());
        extractNormals<_MultiThread>(
            points.data(), normals.data(), (int)points.size());
    }
    void extractVolumeTriMeshs(_Index volumeID, std::vector<_Point>& triPoints,
        std::vector<RGBPixel>& triPixels, std::vector<Point3_<int>>& triIndexs,
        _Float minWeight = 0) const
    {
        const _VolumeData* data = findVolume(volumeID);
        if (!data) return;
        const _VoxelData* neighborData[7] = {0, 0, 0, 0, 0, 0, 0};
        auto getVolumeData = [this](_Index id) -> const _VoxelData* {
            auto* ptr = findVolume(id);
            return ptr ? ptr->voxelData_ : nullptr;
        };
        neighborData[0] = getVolumeData(RIGHTNode(volumeID));
        neighborData[1] = getVolumeData(FRONTNode(volumeID));
        neighborData[2] = getVolumeData(FRONTNode(RIGHTNode(volumeID)));
        neighborData[3] = getVolumeData(TOPNode(volumeID));
        neighborData[4] = getVolumeData(TOPNode(RIGHTNode(volumeID)));
        neighborData[5] = getVolumeData(TOPNode(FRONTNode(volumeID)));
        neighborData[6] =
            getVolumeData(TOPNode(FRONTNode(RIGHTNode(volumeID))));
        std::unordered_map<int, int> pointMap[3];
        _Point originPoint = indexToPoint(volumeID) + (_Float)half_length_;
        ExtractTriMeshImpl<_Type, _Float>::template proc<int>(data->voxelData_,
            _VoxelSize, triPoints, triPixels, triIndexs, originPoint,
            (_Float)unit_length_, minWeight, neighborData, pointMap,
            _VolumeData::getVoxelID);
    }
    void extractTriMeshs(std::vector<_Point>& triPoints,
        std::vector<RGBPixel>& triPixels, std::vector<Point3_<int>>& triIndexs,
        _Float minWeight, std::unordered_map<size_t, int> pointMap[3] = 0) const
    {
        std::unordered_map<size_t, int> nodeMap[3];
        if (!pointMap) pointMap = nodeMap;
        int cnt = 0;
        root_node_.traverseData([this, &triPoints, &triPixels, &triIndexs,
                                    &pointMap, &cnt, minWeight](
                                    _Index ind, const _VolumeData* data) {
            assert(findVolume(ind) == data);
            const _VoxelData* neighborData[7] = {0, 0, 0, 0, 0, 0, 0};
            auto getVolumeData = [this](_Index id) -> const _VoxelData* {
                auto* ptr = findVolume(id);
                return ptr ? ptr->voxelData_ : nullptr;
            };
            neighborData[0] = getVolumeData(RIGHTNode(ind));
            neighborData[1] = getVolumeData(FRONTNode(ind));
            neighborData[2] = getVolumeData(FRONTNode(RIGHTNode(ind)));
            neighborData[3] = getVolumeData(TOPNode(ind));
            neighborData[4] = getVolumeData(TOPNode(RIGHTNode(ind)));
            neighborData[5] = getVolumeData(TOPNode(FRONTNode(ind)));
            neighborData[6] = getVolumeData(TOPNode(FRONTNode(RIGHTNode(ind))));

            Point3_<long long> pointID;
            pointID.x = ((ind & _GETX) >> _SHIFTX) * _VoxelSize;
            pointID.y = ((ind & _GETY) >> _SHIFTY) * _VoxelSize;
            pointID.z = ((ind & _GETZ) >> _SHIFTZ) * _VoxelSize;

            auto originPoint = indexToPoint(ind) + (_Float)half_length_;
            ExtractTriMeshImpl<_Type, _Float>::template proc<size_t>(
                data->voxelData_, _VoxelSize, triPoints, triPixels, triIndexs,
                originPoint, (_Float)unit_length_, minWeight, neighborData,
                pointMap, [&pointID](int x, int y, int z) {
                    size_t c = 0;
                    c |= pointID.x + x;
                    c |= (pointID.y + y) << 21;
                    c |= (pointID.z + z) << 42;
                    return c;
                });
            cnt++;
        });
#ifdef _DEBUG
        // check result
        std::vector<long long> records(triPoints.size(), -1);
        for (int k = 0; k < 3; ++k) {
            for (auto iter = pointMap[k].begin(); iter != pointMap[k].end();
                 ++iter) {
                assert(records[iter->second] == -1);
                records[iter->second] = iter->first;
            }
        }
        for (auto iter = records.begin(); iter != records.end(); ++iter)
            assert(*iter != -1);
#endif
        RGBTSDF_PRINT("extractTriMeshs - Total Number of Volume is %d", cnt);
    }
    const _VoxelData* getTSDFVoxel(_Index id, typename _VolumeData::_Index vox,
        typename _VolumeData::_Index voy,
        typename _VolumeData::_Index voz) const
    {
        assert(vox >= -_VoxelSize && voy >= -_VoxelSize && voz >= -_VoxelSize &&
               vox < (_VoxelSize << 1) && voy < (_VoxelSize << 1) &&
               voz < (_VoxelSize << 1));
        if (vox < 0) { vox += _VoxelSize, id = LEFTNode(id); }
        if (voy < 0) { voy += _VoxelSize, id = BACKNode(id); }
        if (voz < 0) { voz += _VoxelSize, id = BOTTOMNode(id); }
        if (voz >= _VoxelSize) { voz -= _VoxelSize, id = TOPNode(id); }
        if (voy >= _VoxelSize) { voy -= _VoxelSize, id = FRONTNode(id); }
        if (vox >= _VoxelSize) { vox -= _VoxelSize, id = RIGHTNode(id); }
        const _VolumeData* vptr = findVolume(id);
        return vptr ? &(vptr->at(vox, voy, voz)) : nullptr;
    }
    _VoxelData* getTSDFVoxel(_Index id, typename _VolumeData::_Index vox,
        typename _VolumeData::_Index voy, typename _VolumeData::_Index voz)
    {
        using DataNode = OctNode_<0, _MAX_DEPTH, _VolumeData, _Index>;
        const _VoxelData* ptr =
            const_cast<const DataNode*>(this)->getTSDFVoxel(id, vox, voy, voz);
        return const_cast<_VoxelData*>(ptr);
    }
    _Float getTSDFValue(_Index id, typename _VolumeData::_Index vox,
        typename _VolumeData::_Index voy,
        typename _VolumeData::_Index voz) const
    {
        auto* ptr = getTSDFVoxel(id, vox, voy, voz);
        return ptr ? ptr->value() : 1;
    }
    _Float getTSDFValue(const _Point& pt) const
    {
        _Index id = pointToIndex(pt);
        if (id & _INVALID) return 1;
        _Point g   = (pt - indexToPoint(id)) / (_Float)unit_length_;
        auto   idx = (typename _VolumeData::_Index)g.x;
        auto   idy = (typename _VolumeData::_Index)g.y;
        auto   idz = (typename _VolumeData::_Index)g.z;
        g -= _Point {idx, idy, idz} + 0.5;
        if (g.x < 0) idx -= 1, g.x += 1;
        if (g.y < 0) idy -= 1, g.y += 1;
        if (g.z < 0) idz -= 1, g.z += 1;
        _Point b = 1 - g;
        _Float v = b.x * b.y * b.z * getTSDFValue(id, idx, idy, idz);
        v += g.x * b.y * b.z * getTSDFValue(id, idx + 1, idy, idz);
        v += b.x * g.y * b.z * getTSDFValue(id, idx, idy + 1, idz);
        v += g.x * g.y * b.z * getTSDFValue(id, idx + 1, idy + 1, idz);
        v += b.x * b.y * g.z * getTSDFValue(id, idx, idy, idz + 1);
        v += g.x * b.y * g.z * getTSDFValue(id, idx + 1, idy, idz + 1);
        v += b.x * g.y * g.z * getTSDFValue(id, idx, idy + 1, idz + 1);
        v += g.x * g.y * g.z * getTSDFValue(id, idx + 1, idy + 1, idz + 1);
        return v;
    }
    _Point getTSDFNormal(const _Point& pt) const
    {
        _Float nx = getTSDFValue({(_Float)(pt.x + half_length_), pt.y, pt.z}) -
                    getTSDFValue({(_Float)(pt.x - half_length_), pt.y, pt.z});
        _Float ny = getTSDFValue({pt.x, (_Float)(pt.y + half_length_), pt.z}) -
                    getTSDFValue({pt.x, (_Float)(pt.y - half_length_), pt.z});
        _Float nz = getTSDFValue({pt.x, pt.y, (_Float)(pt.z + half_length_)}) -
                    getTSDFValue({pt.x, pt.y, (_Float)(pt.z - half_length_)});
        return _Point(nx, ny, nz).normalize();
    }
    void getTSDFConstraint(const _Point* points,
        TSDFConstraint_<_Float>* constraints, size_t pointNum,
        std::unordered_map<size_t, int> pointMap[3]) const
    {
        // static const _Float _2PI = (_Float)(2 * RGBTSDF_PI);
        if (!points || !constraints || !pointNum) return;
        memset(constraints, 0, pointNum * sizeof(TSDFConstraint_<_Float>));
        for (unsigned char k = 0; k < 3; ++k) {
            constexpr static size_t pointGetX = (0x1 << 21) - 0x1;
            constexpr static size_t pointGetY = pointGetX << 21;
            constexpr static size_t pointGetZ = pointGetY << 21;
            for (auto iter = pointMap[k].begin(); iter != pointMap[k].end();
                 ++iter) {
                // assert(records[iter->second] == -1);
                // records[iter->second] = iter->first;
                size_t pointID   = iter->first;
                size_t pointIDX  = pointID & pointGetX;
                size_t pointIDY  = (pointID & pointGetY) >> 21;
                size_t pointIDZ  = (pointID & pointGetZ) >> 42;
                _Index volumeIDX = (_Index)(pointIDX / _VoxelSize);
                _Index volumeIDY = (_Index)(pointIDY / _VoxelSize);
                _Index volumeIDZ = (_Index)(pointIDZ / _VoxelSize);
                _Index volumeID(0);
                volumeID |= volumeIDX << _SHIFTX;
                volumeID |= volumeIDY << _SHIFTY;
                volumeID |= volumeIDZ << _SHIFTZ;
                Point3_<typename _VolumeData::_Index> voxelID(
                    pointIDX % _VoxelSize, pointIDY % _VoxelSize,
                    pointIDZ % _VoxelSize);
                // auto volumePtr = findVolume(volumeID);
                TSDFConstraint_<_Float>& constraint = constraints[iter->second];
                // assign axis and point value
                constraint.axis  = k;
                constraint.point = indexToPoint(volumeID);
                constraint.point.x +=
                    (_Float)(voxelID.x * unit_length_ + half_length_);
                constraint.point.y +=
                    (_Float)(voxelID.y * unit_length_ + half_length_);
                constraint.point.z +=
                    (_Float)(voxelID.z * unit_length_ + half_length_);
                // first voxel
                const _VoxelData* voxel0 =
                    getTSDFVoxel(volumeID, voxelID.x, voxelID.y, voxelID.z);
                assert(voxel0 != nullptr);
                constraint.val[0] = voxel0->value();
                TSDFPixelGetter<_Type, _Float>::get(voxel0, &constraint.rgb[0]);
                // second voxel
                (&voxelID.x)[k] += 1;
                const _VoxelData* voxel1 =
                    getTSDFVoxel(volumeID, voxelID.x, voxelID.y, voxelID.z);
                assert(voxel1 != nullptr);
                constraint.val[1] = voxel1->value();
                TSDFPixelGetter<_Type, _Float>::get(voxel1, &constraint.rgb[1]);
                // lambda
                constraint.lambda =
                    voxel0->value() / (voxel0->value() - voxel1->value());
#ifdef _DEBUG
                // evaluate point
                const _Point& pt = points[iter->second];
                _Point        offset {0, 0, 0};
                (&offset.x)[constraint.axis] =
                    (_Float)(unit_length_ * constraint.lambda);
                _Point pt_i = constraint.point + offset;
                assert(::abs(pt_i.x - pt.x) < 3e-3 &&
                       ::abs(pt_i.y - pt.y) < 3e-3 &&
                       ::abs(pt_i.z - pt.z) < 3e-3);
#endif
            }
        }
    }
    void computePointPixelGradients(const std::vector<_Point>& points,
        const std::vector<_Point>& normals, const std::vector<_Point>& pixels,
        const std::vector<std::vector<int>>& adjacencyIndexs,
        std::vector<_Point>& gradRs, std::vector<_Point>& gradGs,
        std::vector<_Point>& gradBs) const
    {
        gradRs.resize(points.size());
        gradGs.resize(points.size()), gradBs.resize(points.size());
        memset(gradRs.data(), 0, sizeof(_Point) * gradRs.size());
        memset(gradGs.data(), 0, sizeof(_Point) * gradGs.size());
        memset(gradBs.data(), 0, sizeof(_Point) * gradBs.size());
        for (int i = 0; i < (int)points.size(); ++i) {
            const _Point &pt = points[i], &nl = normals[i];
            const _Point& rgb = pixels[i];
            // compute gradient based on neighbor points
            double                  H[3][3] {}, Br[3] {}, Bg[3] {}, Bb[3] {};
            const std::vector<int>& adjacencyList = adjacencyIndexs[i];
            for (auto iter = adjacencyList.begin(); iter != adjacencyList.end();
                 ++iter) {
                const auto& adj_pt  = points[*iter];
                const auto& adj_rgb = pixels[*iter];

                _Float d = (adj_pt - pt).dot(nl);
                _Float w = (_Float)(unit_length_ / (d + unit_length_));
                w *= w;
                _Point ptv       = (adj_pt - d * nl - pt) * w;
                _Float residualR = ((_Float)adj_rgb.x - rgb.x) * w;
                _Float residualG = ((_Float)adj_rgb.y - rgb.y) * w;
                _Float residualB = ((_Float)adj_rgb.z - rgb.z) * w;
                // clang-format off
                Br[0] += ptv.x*residualR, Br[1] += ptv.y*residualR, Br[2] += ptv.z*residualR;
                Bg[0] += ptv.x*residualG, Bg[1] += ptv.y*residualG, Bg[2] += ptv.z*residualG;
                Bb[0] += ptv.x*residualB, Bb[1] += ptv.y*residualB, Bb[2] += ptv.z*residualB;
                H[0][0] += ptv.x * ptv.x, H[0][1] += ptv.x * ptv.y, H[0][2] += ptv.x * ptv.z;
                H[1][0] += ptv.y * ptv.x, H[1][1] += ptv.y * ptv.y, H[1][2] += ptv.y * ptv.z;
                H[2][0] += ptv.z * ptv.x, H[2][1] += ptv.z * ptv.y, H[2][2] += ptv.z * ptv.z;
                // clang-format on
            }
            // clang-format off
            H[0][0] += nl.x * nl.x, H[0][1] += nl.x * nl.y, H[0][2] += nl.x * nl.z;
            H[1][0] += nl.y * nl.x, H[1][1] += nl.y * nl.y, H[1][2] += nl.y * nl.z;
            H[2][0] += nl.z * nl.x, H[2][1] += nl.z * nl.y, H[2][2] += nl.z * nl.z;
            // clang-format on
            _Point &Gr = gradRs[i], &Gg = gradGs[i], &Gb = gradBs[i];
            // solve $ H * [gx gy gz]' = b $.
            double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) +
                         H[1][0] * (H[2][1] * H[0][2] - H[0][1] * H[2][2]) +
                         H[2][0] * (H[0][1] * H[1][2] - H[1][1] * H[0][2]);
            if ((::abs)(det) < 1e-7 || std::isnan(det)) continue;
            double A11      = (H[1][1] * H[2][2] - H[1][2] * H[2][1]) / det;
            double A12      = (H[1][2] * H[2][0] - H[1][0] * H[2][2]) / det;
            double A13      = (H[1][0] * H[2][1] - H[1][1] * H[2][0]) / det;
            double A21      = (H[0][2] * H[2][1] - H[0][1] * H[2][2]) / det;
            double A22      = (H[0][0] * H[2][2] - H[0][2] * H[2][0]) / det;
            double A23      = (H[0][1] * H[2][0] - H[0][0] * H[2][1]) / det;
            double A31      = (H[0][1] * H[1][2] - H[0][2] * H[1][1]) / det;
            double A32      = (H[0][2] * H[1][0] - H[0][0] * H[1][2]) / det;
            double A33      = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) / det;
            double H_inv[9] = {A11, A21, A31, A12, A22, A32, A13, A23, A33};
            // $ X = [^lambda ^alpha ^theta]' $
            // clang-format off
            Gr.x = (_Float)(H_inv[0] * Br[0] + H_inv[1] * Br[1] + H_inv[2] * Br[2]);
            Gr.y = (_Float)(H_inv[3] * Br[0] + H_inv[4] * Br[1] + H_inv[5] * Br[2]);
            Gr.z = (_Float)(H_inv[6] * Br[0] + H_inv[7] * Br[1] + H_inv[8] * Br[2]);
            Gg.x = (_Float)(H_inv[0] * Bg[0] + H_inv[1] * Bg[1] + H_inv[2] * Bg[2]);
            Gg.y = (_Float)(H_inv[3] * Bg[0] + H_inv[4] * Bg[1] + H_inv[5] * Bg[2]);
            Gb.z = (_Float)(H_inv[6] * Bg[0] + H_inv[7] * Bg[1] + H_inv[8] * Bg[2]);
            Gb.x = (_Float)(H_inv[0] * Bb[0] + H_inv[1] * Bb[1] + H_inv[2] * Bb[2]);
            Gb.y = (_Float)(H_inv[3] * Bb[0] + H_inv[4] * Bb[1] + H_inv[5] * Bb[2]);
            Gb.z = (_Float)(H_inv[6] * Bb[0] + H_inv[7] * Bb[1] + H_inv[8] * Bb[2]);
            // clang-format on
            // test
            assert(!std::isnan(Gr.x) && !std::isnan(Gr.y) && !std::isnan(Gr.z));
            assert(!std::isnan(Gg.x) && !std::isnan(Gg.y) && !std::isnan(Gg.z));
            assert(!std::isnan(Gb.x) && !std::isnan(Gb.y) && !std::isnan(Gb.z));
            std::vector<double> Cr, Cb, Cg;
            for (auto iter = adjacencyList.begin(); iter != adjacencyList.end();
                 ++iter) {
                const auto& adj_pt  = points[*iter];
                const auto& adj_rgb = pixels[*iter];

                const _Point ptv = adj_pt - (adj_pt - pt).dot(nl) * nl - pt;
                const _Float residualR = (_Float)adj_rgb.x - rgb.x;
                const _Float residualG = (_Float)adj_rgb.y - rgb.y;
                const _Float residualB = (_Float)adj_rgb.z - rgb.z;
                Cr.push_back(residualR - ptv.dot(Gr));
                Cg.push_back(residualG - ptv.dot(Gg));
                Cb.push_back(residualB - ptv.dot(Gb));
            }
            // int g = 0;
        }
    }
#ifdef RGBTSDF_USE_EIGEN
    void sovleTSDFConstraint(std::vector<TSDFConstraint_<_Float>>& constraints,
        const std::vector<std::vector<int>>& adjacencyIndexs) const
    {
        // static const _Float _2PI = (_Float)(2 * RGBTSDF_PI);
        std::vector<_Point> points(constraints.size());
        std::vector<_Point> pixels(constraints.size());
        for (size_t i = 0; i < points.size(); ++i) {
            TSDFConstraint_<_Float>::extractPointAndPixel(
                constraints[i], (_Float)unit_length_, points[i], pixels[i]);
        }
        std::vector<_Point> normals(constraints.size());
        extractNormals<false>(
            points.data(), normals.data(), (int)points.size());
        std::vector<_Point> gradRs(constraints.size());
        std::vector<_Point> gradGs(constraints.size());
        std::vector<_Point> gradBs(constraints.size());
        computePointPixelGradients(
            points, normals, pixels, adjacencyIndexs, gradRs, gradGs, gradBs);
        // extractPointAndNormal(constraints.data(), (_Float)unit_length_,
        //    points.data(), normals.data(), points.size());
        int validCnt = 0;
        // using sparse matrix
        using SparseMat  = Eigen::SparseMatrix<double>;
        using TriElement = Eigen::Triplet<double>;
        // double jacobian[3] {}, H[3][3] {}, b[3] {}, residual(0);
        const long long     N = constraints.size();
        std::vector<double> B(N, 0);
        // std::vector<float>  H(N * N, 0);
        std::vector<TriElement> triElements;
        for (int i = 0; i < (int)constraints.size(); ++i) {
            auto&   constraint = constraints[i];
            _Point &pt = points[i], &nl = normals[i];
            // pixel
            _Point        gradRGB[3] = {gradRs[i], gradGs[i], gradBs[i]};
            const _Point& rgb        = pixels[i];
            // calculate data energy and jacobian.
            double measure =
                constraint.val[0] / (constraint.val[0] - constraint.val[1]);
            double residual = (constraint.lambda - measure) * unit_length_;
            double J_i      = unit_length_;
            double JiTJi    = J_i * J_i;
            B[i] += J_i * residual;
            // calculate edge smooth energy and jacobian.
            const std::vector<int>& adjacencyList = adjacencyIndexs[i];
            for (auto iter = adjacencyList.begin(); iter != adjacencyList.end();
                 ++iter) {
                const auto& adj_pt  = points[*iter];
                const auto& adj_rgb = pixels[*iter];
                // $ sin'(x) = cos(x) $
                // $ cos'(x) = sin(x) $
                // $ residual = (C_j.p_0-C_i.p_0)*n +
                // n[C_j.axis]*C_j.lambda*unit - n[C_i.axis]*C_i.lambda*unit $
                double C  = nl.dot(adj_pt - pt);
                double Ji = -(&nl.x)[constraint.axis] * unit_length_;

                JiTJi += Ji * Ji;
                B[i] += Ji * C;
                double Jj    = (&nl.x)[constraints[*iter].axis] * unit_length_;
                double JjTJj = Jj * Jj;
                B[*iter] += Jj * C;
                double JiTJj = Ji * Jj;
                // consider rgb
                const _Point adj_vec = adj_pt - (adj_pt - pt).dot(nl) * nl - pt;
                for (int k = 0; k < 3; ++k) {
                    auto&  Gr = gradRGB[k];
                    double Cr = (&rgb.x)[k] + adj_vec.dot(Gr) - (&adj_rgb.x)[k];
                    // double JRi = (double)(&(constraint.rgb[1].x))[k] -
                    //             (&(constraint.rgb[0].x))[k];
                    double JRi = (&(constraint.rgb[1].x))[k] -
                                 (&(constraint.rgb[0].x))[k];
                    JRi += (&nl.x)[constraint.axis] * unit_length_ * Gr.dot(nl);
                    JRi -= (&Gr.x)[constraint.axis] * unit_length_;
                    JiTJi += JRi * JRi;
                    B[i] += JRi * Cr;
                    // double JRj =
                    // -((double)(&(constraints[*iter].rgb[1].x))[k] -
                    //               (&(constraints[*iter].rgb[0].x))[k]);
                    double JRj = -((&(constraints[*iter].rgb[1].x))[k] -
                                   (&(constraints[*iter].rgb[0].x))[k]);
                    JRj += (&Gr.x)[constraints[*iter].axis] * unit_length_;
                    JRj -= (&nl.x)[constraints[*iter].axis] * unit_length_ *
                           Gr.dot(nl);
                    JjTJj += JRj * JRj;
                    B[*iter] += JRj * Cr;
                    JiTJj += JRi * JRj;
                }
                triElements.emplace_back(i, *iter, JiTJj);
                triElements.emplace_back(*iter, i, JiTJj);
                triElements.emplace_back(*iter, *iter, JjTJj);
                assert(!std::isnan(JiTJj) && !std::isnan(JiTJi));
            }
            triElements.emplace_back(i, i, JiTJi);
            assert(!std::isnan(JiTJi));
            validCnt++;
        }
        SparseMat H(N, N);
        H.setFromTriplets(triElements.begin(), triElements.end());
        Eigen::SimplicialCholesky<SparseMat> chol(H);
        Eigen::Map<Eigen::VectorXd>          b(B.data(), N);
        std::vector<double>                  X(N, 0);
        Eigen::Map<Eigen::VectorXd>(X.data(), N) = chol.solve(b);
        for (int i = 0; i < (int)constraints.size(); ++i) {
            auto& constraint = constraints[i];
            constraint.lambda -= (_Float)X[i];
        }
        RGBTSDF_PRINT(
            "sovleTSDFConstraint - Optimize Point Number: %d.", validCnt);
    }
    void refineTSDFOrientedPoint(_Point* points, RGBPixel* pixels,
        size_t pointNum, const Point3_<int>* tripletIndexs, size_t tripletNum,
        std::unordered_map<size_t, int> pointMap[3], int maxIter = 3) const
    {
        if (maxIter <= 0) return;
        std::vector<std::vector<int>> adjacencyIndexs(pointNum);
        for (size_t i = 0; i < tripletNum; ++i) {
            const auto& triInd = tripletIndexs[i];
            adjacencyIndexs[triInd.x].emplace_back(triInd.y);
            adjacencyIndexs[triInd.x].emplace_back(triInd.z);
            adjacencyIndexs[triInd.y].emplace_back(triInd.x);
            adjacencyIndexs[triInd.y].emplace_back(triInd.z);
            adjacencyIndexs[triInd.z].emplace_back(triInd.x);
            adjacencyIndexs[triInd.z].emplace_back(triInd.y);
        }
        for (size_t i = 0; i < adjacencyIndexs.size(); ++i) {
            std::vector<int>& adjacencyList = adjacencyIndexs[i];
            std::set<int> indexSet(adjacencyList.begin(), adjacencyList.end());
            adjacencyList.assign(indexSet.begin(), indexSet.end());
        }
        // int adjacencyNum =
        //     std::accumulate(adjacencyIndexs.begin(), adjacencyIndexs.end(),
        //     0,
        //         [](int val, const std::vector<int>& iter) {
        //             return (int)iter.size() + val;
        //         });
        std::vector<TSDFConstraint_<_Float>> constraints(pointNum);
        getTSDFConstraint(points, constraints.data(), pointNum, pointMap);
        for (int j = 0; j < maxIter; ++j)
            sovleTSDFConstraint(constraints, adjacencyIndexs);
        // update result
        if (pixels) {
            for (size_t i = 0; i < pointNum; ++i) {
                TSDFConstraint_<_Float>::extractPointAndPixel(
                    constraints[i], (_Float)unit_length_, points[i], pixels[i]);
            }
        } else {
            for (size_t i = 0; i < pointNum; ++i) {
                TSDFConstraint_<_Float>::extractPoint(
                    constraints[i], (_Float)unit_length_, points[i]);
            }
        }
    }
#else
    void refineTSDFOrientedPoint(...) const
    {
        RGBTSDF_PRINT(
            "Include Eigen Header Directory And Add Macro RGBTSDF_USE_EIGEN "
            "Befor rgbtsdf.hpp");
    }
#endif

protected:
    static std::mutex                                         lock_;
    OctNode_<_MAX_DEPTH - 1, _MAX_DEPTH, _VolumeData, _Index> root_node_;
    int                                                       cur_time_;
    const double                                              unit_length_;
    const double                                              half_length_;
    double                                                    trunc_length_;
    const double                                              volume_length_;
    const double                                              octree_length_;
    const Point3_<double>                                     octree_origin_;
};
template <int _MAX_DEPTH, int _VoxelSize, TSDFType _Type, typename _Float,
    class _Index>
std::mutex OcTree_<_MAX_DEPTH, _VoxelSize, _Type, _Float, _Index>::lock_;

template <TSDFType _Type, typename _Float> struct InitVoxelImpl {
    static inline void proc(TSDFVoxel_<_Type, _Float>* voxelData, int voxelNum)
    {
        assert(voxelData && voxelNum > 0);
        TSDFVoxel_<_Type, _Float>* cur = voxelData;
        for (; cur != voxelData + voxelNum; ++cur)
            *cur = TSDFDefaultValue<_Type, _Float>::value;
    }
};

template <TSDFType _Type, typename _Float, bool _Inverse> struct IntegrateImpl {
    static inline void proc(TSDFVoxel_<_Type, _Float>* voxelData, int voxelSize,
        const K_<_Float>& cam, const RT_<_Float>& RT,
        coord_trait_t<_Float> nearZ, coord_trait_t<_Float> farZ,
        const Point3_<_Float>& originPoint, coord_trait_t<_Float> voxelUnit,
        coord_trait_t<_Float> truncUnit, int curTime,
        const InArr_<_Float>& depth, const InArr_<RGBPixel>& color,
        const InArr_<_Float>& i2w = nullptr)
    {
        const static RGBPixel dark = {0, 0, 0};
        assert(!depth.empty() && voxelData && voxelUnit > 0);
#ifdef RGBTSDF_RESTRAIN_THIN_OBJECT
        const _Float unitSquare = voxelUnit * voxelUnit;
#endif
        const bool useColor = !color.empty() && color.size() == depth.size();
        if (!TSDFPixelGetter<_Type, _Float>::check(useColor)) return;
        const bool useWeight = !i2w.empty() && i2w.width == depth.width &&
                               i2w.height == depth.height;
        const auto img_w1 = (_Float)depth.width - 1;
        const auto img_h1 = (_Float)depth.height - 1;
        const auto ptx    = Point3_<_Float>(RT.a1, RT.b1, RT.c1) * voxelUnit;
        const auto pty    = Point3_<_Float>(RT.a2, RT.b2, RT.c2) * voxelUnit;
        const auto ptz    = Point3_<_Float>(RT.a3, RT.b3, RT.c3) * voxelUnit;
        const auto pto    = RT.transformPoint(originPoint + voxelUnit / 2);
        _Float     zx = pto.x, zy = pto.y, zz = pto.z;
        TSDFVoxel_<_Type, _Float>* voxelPtr = voxelData;
        for (int z = 0; z < voxelSize;
             ++z, zx += ptz.x, zy += ptz.y, zz += ptz.z) {
            _Float yx = zx, yy = zy, yz = zz;
            for (int y = 0; y < voxelSize;
                 ++y, yx += pty.x, yy += pty.y, yz += pty.z) {
                _Float xx = yx, xy = yy, xz = yz;
                for (int x = 0; x < voxelSize;
                     ++x, ++voxelPtr, xx += ptx.x, xy += ptx.y, xz += ptx.z) {
                    if (xz <= 0) continue;
                    _Float u = xx * cam.fx / xz + cam.cx;
                    _Float v = xy * cam.fy / xz + cam.cy;
                    if (!(u >= 0 && v >= 0 && u < img_w1 && v < img_h1))
                        continue;
                    _Float d = 0, dw = 0;
                    int    ui = (int)u, vi = (int)v;
                    // change offset by stride
                    const _Float &d00 = depth.ptr(vi)[ui], &d01 = (&d00)[1];
                    const _Float& d10 =
                        *(const _Float*)((const char*)&d00 + depth.stride);
                    const _Float& d11 = (&d10)[1];

                    _Float b0 = u - (_Float)ui, a0 = 1.0f - b0;
                    _Float b1 = v - (_Float)vi, a1 = 1.0f - b1;
                    // clang-format off
                    if (d00 > nearZ && d00 < farZ) { _Float t = a1 * a0; d += t * d00, dw += t; }
                    if (d01 > nearZ && d01 < farZ) { _Float t = a1 * b0; d += t * d01, dw += t; }
                    if (d10 > nearZ && d10 < farZ) { _Float t = b1 * a0; d += t * d10, dw += t; }
                    if (d11 > nearZ && d11 < farZ) { _Float t = b1 * b0; d += t * d11, dw += t; }
                    // clang-format on
                    if (dw == 0) continue;
                    _Float dist = std::sqrt(xx * xx + xy * xy + xz * xz);
                    _Float z2d  = dist / xz;
                    _Float sdf  = (d / dw - xz) * z2d;
                    if (sdf >= -truncUnit) {
#ifdef RGBTSDF_RESTRAIN_THIN_OBJECT
                        // penally function, restrain thin structure.
                        _Float dist01 = (d01 - d00) * z2d;
                        _Float dist02 = (d10 - d00) * z2d;
                        _Float dist03 = (d11 - d00) * z2d;
                        dw *= std::sqrt(unitSquare /
                                        (unitSquare + dist01 * dist01 +
                                            dist02 * dist02 + dist03 * dist03));
#endif
                        TSDFValueSetter<_Type, _Float, _Inverse>::add(*voxelPtr,
                            sdf < truncUnit ? sdf / truncUnit : 1,
                            useWeight ? i2w.ptr(vi)[ui] * dw : dw,
                            useColor ? color.ptr(vi)[ui] : dark, curTime);
                    }
                }
            }
        }
    }
};

template <TSDFType _Type, typename _Float> struct ExtractVoxelImpl {
    static inline void proc(const TSDFVoxel_<_Type, _Float>* voxelData,
        int voxelSize, std::vector<Point3_<_Float>>& points,
        std::vector<RGBPixel>& pixels, Point3_<_Float> originPoint,
        coord_trait_t<_Float> unitLength, coord_trait_t<_Float> minWeight)
    {
        assert(voxelData && voxelSize > 0);
        originPoint += unitLength / 2;
        std::vector<_Float> x_vec(voxelSize), y_vec(voxelSize);
        for (int i = 0; i < voxelSize; ++i) {
            x_vec[i] = i * unitLength + originPoint.x;
            y_vec[i] = i * unitLength + originPoint.y;
        }
        const TSDFVoxel_<_Type, _Float>* voxelPtr = voxelData;
        for (int z = 0; z < voxelSize; ++z) {
            _Float z0 = z * unitLength + originPoint.z;
            for (int y = 0; y < voxelSize; ++y) {
                _Float y0 = y_vec[y];
                for (int x = 0; x < voxelSize; ++x, ++voxelPtr) {
                    _Float x0 = x_vec[x];
                    if (voxelPtr->w > minWeight && std::abs(voxelPtr->v) < 1) {
                        points.emplace_back(x0, y0, z0);
                        if (voxelPtr->v < 0x0) {
                            unsigned char b = (unsigned char)clamp<_Float>(
                                std::abs(voxelPtr->v) * 0xFF, 0x0, 0xFF);
                            unsigned char g = (unsigned char)(0xFF - b);
                            pixels.emplace_back(0x0, g, b);
                        } else {
                            unsigned char r = (unsigned char)clamp<_Float>(
                                voxelPtr->v * 0xFF, 0x0, 0xFF);
                            unsigned char g = (unsigned char)(0xFF - r);
                            pixels.emplace_back(r, g, 0x0);
                        }
                    }
                }
            }
        }
    }
};

template <TSDFType _Type, typename _Float> struct ExtractPointImpl {
    typedef TSDFVoxel_<_Type, _Float> _TSDFVoxel;
    // Interpolate Formula Is $P=P1+(isovalue-V1)*(P2-P1)/(V2-V1)$
    // Consider ISOValue Is Zero In TSDF, So $P=(P1*V2-P2*V1)/(V2-V1)$
    static inline void interp(const Point3_<_Float>& p0, const _TSDFVoxel* cur,
        const _TSDFVoxel* top, const _TSDFVoxel* front, const _TSDFVoxel* right,
        _Float unitLength, _Float minWeight,
        std::vector<Point3_<_Float>>& points, std::vector<RGBPixel>& pixels)
    {
        _Float curTsdf = (std::abs)(cur->value());
        if (!(cur->weight() > minWeight && curTsdf < 1)) return;
        _Float curTsdfUnit = curTsdf * unitLength;
        if (top && top->weight() > minWeight &&
            NotSameSign(cur->value(), top->value()) &&
            (std::abs)(top->value()) < 1) {
            points.emplace_back(p0.x, p0.y,
                p0.z + curTsdfUnit / (curTsdf + (std::abs)(top->value())));
            TSDFPixelGetter<_Type, _Float>::interp(cur, top, &pixels);
        }
        if (front && front->weight() > minWeight &&
            NotSameSign(cur->value(), front->value()) &&
            (std::abs)(front->value()) < 1) {
            points.emplace_back(p0.x,
                p0.y + curTsdfUnit / (curTsdf + (std::abs)(front->value())),
                p0.z);
            TSDFPixelGetter<_Type, _Float>::interp(cur, front, &pixels);
        }
        if (right && right->weight() > minWeight &&
            NotSameSign(cur->value(), right->value()) &&
            (std::abs)(right->value()) < 1) {
            points.emplace_back(
                p0.x + curTsdfUnit / (curTsdf + (std::abs)(right->value())),
                p0.y, p0.z);
            TSDFPixelGetter<_Type, _Float>::interp(cur, right, &pixels);
        }
    }
    static inline void proc(const _TSDFVoxel* voxelData, int voxelSize,
        std::vector<Point3_<_Float>>& points, std::vector<RGBPixel>& pixels,
        Point3_<_Float> originPoint, coord_trait_t<_Float> unitLength,
        coord_trait_t<_Float> minWeight, const _TSDFVoxel* right,
        const _TSDFVoxel* front, const _TSDFVoxel* top)
    {
        const int prevSize   = voxelSize - 1;
        const int volumeArea = voxelSize * voxelSize;
        assert(voxelData && voxelSize > 0);
        _Float              z0(originPoint.z + unitLength / 2);
        std::vector<_Float> xvec(voxelSize), yvec(voxelSize);
        xvec[0] = originPoint.x + unitLength / 2;
        for (auto cur = xvec.begin() + 1; cur != xvec.end(); ++cur)
            *cur = cur[-1] + unitLength;
        yvec[0] = originPoint.y + unitLength / 2;
        for (auto cur = yvec.begin() + 1; cur != yvec.end(); ++cur)
            *cur = cur[-1] + unitLength;
        const _TSDFVoxel* voxelPtr = voxelData;
        const _TSDFVoxel *rightPtr = right, *frontPtr = front;
        for (int z = 0; z < prevSize;
             ++z, z0 += unitLength, frontPtr += volumeArea) {
            for (int y = 0; y < prevSize; ++y, rightPtr += voxelSize) {
                for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
                    interp({xvec[x], yvec[y], z0}, voxelPtr,
                        voxelPtr + volumeArea, voxelPtr + voxelSize,
                        voxelPtr + 1, unitLength, minWeight, points, pixels);
                }
                // consider x == prev_size
                interp({xvec[prevSize], yvec[y], z0}, voxelPtr,
                    voxelPtr + volumeArea, voxelPtr + voxelSize,
                    right ? rightPtr : nullptr, unitLength, minWeight, points,
                    pixels);
                voxelPtr++;
            }
            // consider y == prev_size
            for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
                interp({xvec[x], yvec[prevSize], z0}, voxelPtr,
                    voxelPtr + volumeArea, front ? frontPtr + x : nullptr,
                    voxelPtr + 1, unitLength, minWeight, points, pixels);
            }
            // consider x == prev_size
            interp({xvec[prevSize], yvec[prevSize], z0}, voxelPtr,
                voxelPtr + volumeArea, front ? frontPtr + prevSize : nullptr,
                right ? rightPtr : nullptr, unitLength, minWeight, points,
                pixels);
            voxelPtr++;
            rightPtr += voxelSize;
        }
        // consider z == prev_size
        const _TSDFVoxel* topPtr = top;
        for (int y = 0; y < prevSize;
             ++y, topPtr += voxelSize, rightPtr += voxelSize) {
            for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
                interp({xvec[x], yvec[y], z0}, voxelPtr,
                    top ? topPtr + x : nullptr, voxelPtr + voxelSize,
                    voxelPtr + 1, unitLength, minWeight, points, pixels);
            }
            // consider x == prev_size
            interp({xvec[prevSize], yvec[y], z0}, voxelPtr,
                top ? topPtr + prevSize : nullptr, voxelPtr + voxelSize,
                right ? rightPtr : nullptr, unitLength, minWeight, points,
                pixels);
            voxelPtr++;
        }
        // consider y == prev_size
        for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
            interp({xvec[x], yvec[prevSize], z0}, voxelPtr,
                top ? topPtr + x : nullptr, front ? frontPtr + x : nullptr,
                voxelPtr + 1, unitLength, minWeight, points, pixels);
        }
        // consider x == prev_size
        interp({xvec[prevSize], yvec[prevSize], z0}, voxelPtr,
            top ? topPtr + prevSize : nullptr,
            front ? frontPtr + prevSize : nullptr, right ? rightPtr : nullptr,
            unitLength, minWeight, points, pixels);
    }
};

extern const int edgeTable[256];
extern const int triTable[256][16];
template <TSDFType _Type, typename _Float> struct ExtractTriMeshImpl {
    typedef TSDFVoxel_<_Type, _Float> _TSDFVoxel;
    // implementation for marching cubes, based on
    // https://paulbourke.net/geometry/polygonise/
    template <typename _Index = int> static inline void interp(
        const Point3_<_Float>& p, const _TSDFVoxel& v0, const _TSDFVoxel& v1,
        const _TSDFVoxel& v2, const _TSDFVoxel& v3, const _TSDFVoxel& v4,
        const _TSDFVoxel& v5, const _TSDFVoxel& v6, const _TSDFVoxel& v7,
        _Float unit, _Float minWeight, std::vector<Point3_<_Float>>& triPoints,
        std::vector<RGBPixel>& triPixels, std::vector<Point3_<int>>& triIndexs,
        std::unordered_map<_Index, int>      pointMap[3],
        std::function<_Index(int, int, int)> getPointID)
    {
        int vertinds[12];
        if (v0.weight() <= minWeight || v1.weight() <= minWeight ||
            v2.weight() <= minWeight || v3.weight() <= minWeight ||
            v4.weight() <= minWeight || v5.weight() <= minWeight ||
            v6.weight() <= minWeight || v7.weight() <= minWeight)
            return;
        int cubeindex = 0;
        if (v0.value() < 0) cubeindex |= 0x01;
        if (v1.value() < 0) cubeindex |= 0x02;
        if (v2.value() < 0) cubeindex |= 0x04;
        if (v3.value() < 0) cubeindex |= 0x08;
        if (v4.value() < 0) cubeindex |= 0x10;
        if (v5.value() < 0) cubeindex |= 0x20;
        if (v6.value() < 0) cubeindex |= 0x40;
        if (v7.value() < 0) cubeindex |= 0x80;
        if (edgeTable[cubeindex] == 0x0) return;
        _Index pointID   = getPointID(0, 0, 0);
        _Index pointIDX  = getPointID(1, 0, 0);
        _Index pointIDY  = getPointID(0, 1, 0);
        _Index pointIDZ  = getPointID(0, 0, 1);
        _Index pointIDXY = getPointID(1, 1, 0);
        _Index pointIDXZ = getPointID(1, 0, 1);
        _Index pointIDYZ = getPointID(0, 1, 1);
        if (edgeTable[cubeindex] & 0x001) {
            if (pointMap[0].find(pointID) == pointMap[0].end()) {
                vertinds[0]          = (int)triPoints.size();
                pointMap[0][pointID] = (int)triPoints.size();
                triPoints.emplace_back(
                    p.x - v0.value() * unit / (v1.value() - v0.value()), p.y,
                    p.z);
                TSDFPixelGetter<_Type, _Float>::interp(&v0, &v1, &triPixels);
            } else {
                vertinds[0] = pointMap[0][pointID];
            }
        }
        if (edgeTable[cubeindex] & 0x002) {
            if (pointMap[1].find(pointIDX) == pointMap[1].end()) {
                vertinds[1]           = (int)triPoints.size();
                pointMap[1][pointIDX] = (int)triPoints.size();
                triPoints.emplace_back(p.x + unit,
                    p.y - v1.value() * unit / (v2.value() - v1.value()), p.z);
                TSDFPixelGetter<_Type, _Float>::interp(&v1, &v2, &triPixels);
            } else {
                vertinds[1] = pointMap[1][pointIDX];
            }
        }
        if (edgeTable[cubeindex] & 0x004) {
            if (pointMap[0].find(pointIDY) == pointMap[0].end()) {
                vertinds[2]           = (int)triPoints.size();
                pointMap[0][pointIDY] = (int)triPoints.size();
                triPoints.emplace_back(
                    p.x - v3.value() * unit / (v2.value() - v3.value()),
                    p.y + unit, p.z);
                TSDFPixelGetter<_Type, _Float>::interp(&v2, &v3, &triPixels);
            } else {
                vertinds[2] = pointMap[0][pointIDY];
            }
        }
        if (edgeTable[cubeindex] & 0x008) {
            if (pointMap[1].find(pointID) == pointMap[1].end()) {
                pointMap[1][pointID] = (int)triPoints.size();
                vertinds[3]          = (int)triPoints.size();
                triPoints.emplace_back(p.x,
                    p.y - v0.value() * unit / (v3.value() - v0.value()), p.z);
                TSDFPixelGetter<_Type, _Float>::interp(&v0, &v3, &triPixels);
            } else {
                vertinds[3] = pointMap[1][pointID];
            }
        }
        if (edgeTable[cubeindex] & 0x010) {
            if (pointMap[0].find(pointIDZ) == pointMap[0].end()) {
                pointMap[0][pointIDZ] = (int)triPoints.size();
                vertinds[4]           = (int)triPoints.size();
                triPoints.emplace_back(
                    p.x - v4.value() * unit / (v5.value() - v4.value()), p.y,
                    p.z + unit);
                TSDFPixelGetter<_Type, _Float>::interp(&v4, &v5, &triPixels);
            } else {
                vertinds[4] = pointMap[0][pointIDZ];
            }
        }
        if (edgeTable[cubeindex] & 0x020) {
            if (pointMap[1].find(pointIDXZ) == pointMap[1].end()) {
                pointMap[1][pointIDXZ] = (int)triPoints.size();
                vertinds[5]            = (int)triPoints.size();
                triPoints.emplace_back(p.x + unit,
                    p.y - v5.value() * unit / (v6.value() - v5.value()),
                    p.z + unit);
                TSDFPixelGetter<_Type, _Float>::interp(&v5, &v6, &triPixels);
            } else {
                vertinds[5] = pointMap[1][pointIDXZ];
            }
        }
        if (edgeTable[cubeindex] & 0x040) {
            if (pointMap[0].find(pointIDYZ) == pointMap[0].end()) {
                pointMap[0][pointIDYZ] = (int)triPoints.size();
                vertinds[6]            = (int)triPoints.size();
                triPoints.emplace_back(
                    p.x - v7.value() * unit / (v6.value() - v7.value()),
                    p.y + unit, p.z + unit);
                TSDFPixelGetter<_Type, _Float>::interp(&v6, &v7, &triPixels);
            } else {
                vertinds[6] = pointMap[0][pointIDYZ];
            }
        }
        if (edgeTable[cubeindex] & 0x080) {
            if (pointMap[1].find(pointIDZ) == pointMap[1].end()) {
                pointMap[1][pointIDZ] = (int)triPoints.size();
                vertinds[7]           = (int)triPoints.size();
                triPoints.emplace_back(p.x,
                    p.y - v4.value() * unit / (v7.value() - v4.value()),
                    p.z + unit);
                TSDFPixelGetter<_Type, _Float>::interp(&v4, &v7, &triPixels);
            } else {
                vertinds[7] = pointMap[1][pointIDZ];
            }
        }
        if (edgeTable[cubeindex] & 0x100) {
            if (pointMap[2].find(pointID) == pointMap[2].end()) {
                pointMap[2][pointID] = (int)triPoints.size();
                vertinds[8]          = (int)triPoints.size();
                triPoints.emplace_back(p.x, p.y,
                    p.z - v0.value() * unit / (v4.value() - v0.value()));
                TSDFPixelGetter<_Type, _Float>::interp(&v0, &v4, &triPixels);
            } else {
                vertinds[8] = pointMap[2][pointID];
            }
        }
        if (edgeTable[cubeindex] & 0x200) {
            if (pointMap[2].find(pointIDX) == pointMap[2].end()) {
                pointMap[2][pointIDX] = (int)triPoints.size();
                vertinds[9]           = (int)triPoints.size();
                triPoints.emplace_back(p.x + unit, p.y,
                    p.z - v1.value() * unit / (v5.value() - v1.value()));
                TSDFPixelGetter<_Type, _Float>::interp(&v1, &v5, &triPixels);
            } else {
                vertinds[9] = pointMap[2][pointIDX];
            }
        }
        if (edgeTable[cubeindex] & 0x400) {
            if (pointMap[2].find(pointIDXY) == pointMap[2].end()) {
                pointMap[2][pointIDXY] = (int)triPoints.size();
                vertinds[10]           = (int)triPoints.size();
                triPoints.emplace_back(p.x + unit, p.y + unit,
                    p.z - v2.value() * unit / (v6.value() - v2.value()));
                TSDFPixelGetter<_Type, _Float>::interp(&v2, &v6, &triPixels);
            } else {
                vertinds[10] = pointMap[2][pointIDXY];
            }
        }
        if (edgeTable[cubeindex] & 0x800) {
            if (pointMap[2].find(pointIDY) == pointMap[2].end()) {
                pointMap[2][pointIDY] = (int)triPoints.size();
                vertinds[11]          = (int)triPoints.size();
                triPoints.emplace_back(p.x, p.y + unit,
                    p.z - v3.value() * unit / (v7.value() - v3.value()));
                TSDFPixelGetter<_Type, _Float>::interp(&v3, &v7, &triPixels);
            } else {
                vertinds[11] = pointMap[2][pointIDY];
            }
        }
        for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
            triIndexs.emplace_back(vertinds[triTable[cubeindex][i]],
                vertinds[triTable[cubeindex][i + 2]],
                vertinds[triTable[cubeindex][i + 1]]);
        }
    }
    template <typename _Index>
    static inline void proc(const _TSDFVoxel* voxelData, int voxelSize,
        std::vector<Point3_<_Float>>& triPoints,
        std::vector<RGBPixel>& triPixels, std::vector<Point3_<int>>& triIndexs,
        Point3_<_Float> originPoint, coord_trait_t<_Float> unitLength,
        coord_trait_t<_Float> minWeight, const _TSDFVoxel* neighborData[7],
        std::unordered_map<_Index, int>      pointMap[3],
        std::function<_Index(int, int, int)> getPointID)
    {
        const int prevSize    = voxelSize - 1;
        const int _VolumeArea = voxelSize * voxelSize;
        const int _ID0        = 0;
        const int _ID1        = 1;
        const int _ID2        = voxelSize + 1;
        const int _ID3        = voxelSize;
        const int _ID4        = _VolumeArea;
        const int _ID5        = _VolumeArea + 1;
        const int _ID6        = _VolumeArea + voxelSize + 1;
        const int _ID7        = _VolumeArea + voxelSize;
        // auto getPointID = TSDFVolume_<_Type, _Float, _VoxelSize>::getVoxelID;
        assert(voxelData);
        _Float              z0(originPoint.z);
        std::vector<_Float> xvec(voxelSize), yvec(voxelSize);
        xvec[0] = originPoint.x;
        for (_Float* cur = &xvec[1]; cur != &xvec[0] + voxelSize; ++cur)
            *cur = cur[-1] + unitLength;
        yvec[0] = originPoint.y;
        for (_Float* cur = &yvec[1]; cur != &yvec[0] + voxelSize; ++cur)
            *cur = cur[-1] + unitLength;
        const _TSDFVoxel* voxelPtr         = voxelData;
        const _TSDFVoxel* rightPtr         = neighborData[0];
        const _TSDFVoxel* frontPtr         = neighborData[1];
        const _TSDFVoxel* frontRightPtr    = neighborData[2];
        const _TSDFVoxel* topPtr           = neighborData[3];
        const _TSDFVoxel* topRightPtr      = neighborData[4];
        const _TSDFVoxel* topFrontPtr      = neighborData[5];
        const _TSDFVoxel* topFrontRightPtr = neighborData[6];
        for (int z = 0; z < prevSize; ++z, z0 += unitLength,
                 frontPtr += _VolumeArea, frontRightPtr += _VolumeArea) {
            for (int y = 0; y < prevSize; ++y, rightPtr += voxelSize) {
                for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
                    interp<_Index>({xvec[x], yvec[y], z0}, voxelPtr[_ID0],
                        voxelPtr[_ID1], voxelPtr[_ID2], voxelPtr[_ID3],
                        voxelPtr[_ID4], voxelPtr[_ID5], voxelPtr[_ID6],
                        voxelPtr[_ID7], unitLength, minWeight, triPoints,
                        triPixels, triIndexs, pointMap,
                        [x, y, z, getPointID](int xi, int yi, int zi) {
                            return getPointID(x + xi, y + yi, z + zi);
                        });
                }
                // Consider x == prevSize
                auto& voxel = *voxelPtr++;
                if (!neighborData[0]) continue;
                interp<_Index>({xvec[prevSize], yvec[y], z0}, (&voxel)[_ID0],
                    rightPtr[_ID0], rightPtr[_ID3], (&voxel)[_ID3],
                    (&voxel)[_ID4], rightPtr[_ID4], rightPtr[_ID7],
                    (&voxel)[_ID7], unitLength, minWeight, triPoints, triPixels,
                    triIndexs, pointMap,
                    [prevSize, y, z, getPointID](int xi, int yi, int zi) {
                        return getPointID(prevSize + xi, y + yi, z + zi);
                    });
            }
            // Consider y == prevSize
            if (!neighborData[1]) {
                voxelPtr += voxelSize;
                rightPtr += voxelSize;
                continue;
            }
            for (int x = 0; x < prevSize; ++x, ++voxelPtr) {
                interp<_Index>({xvec[x], yvec[prevSize], z0}, voxelPtr[_ID0],
                    voxelPtr[_ID1], frontPtr[x + _ID1], frontPtr[x + _ID0],
                    voxelPtr[_ID4], voxelPtr[_ID5], frontPtr[x + _ID5],
                    frontPtr[x + _ID4], unitLength, minWeight, triPoints,
                    triPixels, triIndexs, pointMap,
                    [x, prevSize, z, getPointID](int xi, int yi, int zi) {
                        return getPointID(x + xi, prevSize + yi, z + zi);
                    });
            }
            // Consider x == prevSize
            auto& voxel = *voxelPtr++;
            if (!neighborData[0] || !neighborData[2]) {
                rightPtr += voxelSize;
                continue;
            }
            interp<_Index>({xvec[prevSize], yvec[prevSize], z0}, (&voxel)[_ID0],
                rightPtr[_ID0], frontRightPtr[_ID0], frontPtr[prevSize],
                (&voxel)[_ID4], rightPtr[_ID4], frontRightPtr[_ID4],
                frontPtr[_VolumeArea + prevSize], unitLength, minWeight,
                triPoints, triPixels, triIndexs, pointMap,
                [prevSize, z, getPointID](int xi, int yi, int zi) {
                    return getPointID(prevSize + xi, prevSize + yi, z + zi);
                });
            rightPtr += voxelSize;
        }
        // Consider z == prevSize
        if (!neighborData[3]) return;
        for (int y = 0; y < prevSize;
             ++y, rightPtr += voxelSize, topRightPtr += voxelSize) {
            for (int x = 0; x < prevSize; ++x, ++voxelPtr, ++topPtr) {
                interp<_Index>({xvec[x], yvec[y], z0}, voxelPtr[_ID0],
                    voxelPtr[_ID1], voxelPtr[_ID2], voxelPtr[_ID3],
                    topPtr[_ID0], topPtr[_ID1], topPtr[_ID2], topPtr[_ID3],
                    unitLength, minWeight, triPoints, triPixels, triIndexs,
                    pointMap,
                    [x, y, prevSize, getPointID](int xi, int yi, int zi) {
                        return getPointID(x + xi, y + yi, prevSize + zi);
                    });
            }
            // Consider x == prevSize
            auto& top   = *topPtr++;
            auto& voxel = *voxelPtr++;
            if (!neighborData[0] || !neighborData[4]) continue;
            interp<_Index>({xvec[prevSize], yvec[y], z0}, (&voxel)[_ID0],
                rightPtr[_ID0], rightPtr[_ID3], (&voxel)[_ID3], (&top)[_ID0],
                topRightPtr[_ID0], topRightPtr[_ID3], (&top)[_ID3], unitLength,
                minWeight, triPoints, triPixels, triIndexs, pointMap,
                [prevSize, y, getPointID](int xi, int yi, int zi) {
                    return getPointID(prevSize + xi, y + yi, prevSize + zi);
                });
        }
        // Consider y == prevSize
        if (!neighborData[5] || !neighborData[1]) return;
        for (int x = 0; x < prevSize;
             ++x, ++voxelPtr, ++topPtr, ++topFrontPtr) {
            interp<_Index>({xvec[x], yvec[prevSize], z0}, voxelPtr[_ID0],
                voxelPtr[_ID1], frontPtr[x + _ID1], frontPtr[x + _ID0],
                topPtr[_ID0], topPtr[_ID1], topFrontPtr[_ID1],
                topFrontPtr[_ID0], unitLength, minWeight, triPoints, triPixels,
                triIndexs, pointMap,
                [x, prevSize, getPointID](int xi, int yi, int zi) {
                    return getPointID(x + xi, prevSize + yi, prevSize + zi);
                });
        }
        // Consider x == prevSize
        if (!neighborData[0] || !neighborData[2] || !neighborData[4] ||
            !neighborData[6])
            return;
        interp<_Index>({xvec[prevSize], yvec[prevSize], z0}, voxelPtr[_ID0],
            rightPtr[_ID0], frontRightPtr[_ID0], frontPtr[prevSize],
            topPtr[_ID0], topRightPtr[_ID0], topFrontRightPtr[_ID0],
            topFrontPtr[_ID0], unitLength, minWeight, triPoints, triPixels,
            triIndexs, pointMap,
            [prevSize, getPointID](int xi, int yi, int zi) {
                return getPointID(prevSize + xi, prevSize + yi, prevSize + zi);
            });
    }
};

#if defined(_WIN32) && defined(_MSC_VER)
#define RGBTSDF_FCLOSE fclose
#define RGBTSDF_FFLUSH fflush
#define RGBTSDF_FSCANF fscanf_s
#define RGBTSDF_FPRINTF fprintf_s
#define RGBTSDF_FOPEN(file, name, mode) fopen_s(&file, name, mode)
#else
#define RGBTSDF_FCLOSE fclose
#define RGBTSDF_FFLUSH fflush
#define RGBTSDF_FSCANF fscanf
#define RGBTSDF_FPRINTF fprintf
#define RGBTSDF_FOPEN(file, name, mode) file = fopen(name, mode)
#endif

template <typename _Float = float> static inline bool writePointsToASC(
    const char* ascPath, const _Float pointData[][3], size_t pointNum)
{
    if (!pointData) return false;
    FILE* f = nullptr;
    RGBTSDF_FOPEN(f, ascPath, "w");
    if (!f) return false;
    for (size_t i = 0; i < pointNum; ++i) {
        auto& pt = pointData[i];
        RGBTSDF_FPRINTF(f, "%f %f %f\n", pt[0], pt[1], pt[2]);
    }
    RGBTSDF_FFLUSH(f);
    return RGBTSDF_FCLOSE(f) == 0;
}
template <typename _Float = float> static inline bool writePointsToASC(
    const char* ascPath, const _Float pointData[][3],
    const coord_trait_t<_Float> normalData[][3], size_t pointNum)
{
    if (!normalData) return writePointsToASC(ascPath, pointData, pointNum);
    FILE* f = nullptr;
    RGBTSDF_FOPEN(f, ascPath, "w");
    if (!f) return false;
    for (size_t i = 0; i < pointNum; ++i) {
        auto pt = pointData[i], nl = normalData[i];
        RGBTSDF_FPRINTF(
            f, "%f %f %f %f %f %f\n", pt[0], pt[1], pt[2], nl[0], nl[1], nl[2]);
    }
    RGBTSDF_FFLUSH(f);
    return RGBTSDF_FCLOSE(f) == 0;
}
template <typename _Float = float>
static inline bool writePointsToASC(const char* ascPath,
    const _Float pointData[][3], const coord_trait_t<_Float> normalData[][3],
    const unsigned char rgbData[][3], size_t pointNum)
{
    if (!rgbData)
        return writePointsToASC(ascPath, pointData, normalData, pointNum);
    if (!normalData) return writePointsToASC(ascPath, pointData, pointNum);
    FILE* f = nullptr;
    RGBTSDF_FOPEN(f, ascPath, "w");
    if (!f) return false;
    for (size_t i = 0; i < pointNum; ++i) {
        auto rgb = rgbData[i];
        auto pt = pointData[i], nl = normalData[i];
#ifndef RGBTSDF_USE_BGR_COLOR
        RGBTSDF_FPRINTF(f, "%f %f %f %f %f %f %d %d %d\n", pt[0], pt[1], pt[2],
            nl[0], nl[1], nl[2], rgb[0], rgb[1], rgb[2]);
#else
        RGBTSDF_FPRINTF(f, "%f %f %f %f %f %f %d %d %d\n", pt[0], pt[1], pt[2],
            nl[0], nl[1], nl[2], rgb[2], rgb[1], rgb[0]);
#endif
    }
    RGBTSDF_FFLUSH(f);
    return RGBTSDF_FCLOSE(f) == 0;
}
template <typename _Float = float> static inline bool writePointsToASC(
    const char* ascPath, const Point3_<_Float>* pointData, size_t pointNum)
{
    return writePointsToASC(ascPath, (const _Float(*)[3])pointData, pointNum);
}
template <typename _Float = float> static inline bool writePointsToASC(
    const char* ascPath, const Point3_<_Float>* pointData,
    const Point3_<coord_trait_t<_Float>>* normalData, size_t pointNum)
{
    return writePointsToASC(ascPath, (const _Float(*)[3])pointData,
        (const _Float(*)[3])normalData, pointNum);
}
template <typename _Float = float> static inline bool writePointsToASC(
    const char* ascPath, const Point3_<_Float>* pointData,
    const Point3_<coord_trait_t<_Float>>* normalData, const RGBPixel* rgbData,
    size_t pointNum)
{
    return writePointsToASC(ascPath, (const _Float(*)[3])pointData,
        (const _Float(*)[3])normalData, (const unsigned char(*)[3])rgbData,
        pointNum);
}
template <typename _Float, typename _Index>
bool writeTriMeshToOBJ(const char* meshPath, const _Float triPoints[][3],
    const _Float triNormals[][3], const unsigned char triPixels[][3],
    size_t pointNum, const _Index triIndexs[][3], size_t indexNum)
{
    FILE* f = nullptr;
    if (!(triPoints != nullptr && pointNum > 0 && triIndexs != nullptr &&
            indexNum > 0))
        return false;
    RGBTSDF_FOPEN(f, meshPath, "w");
    if (!f) return false;
    if (triPixels != nullptr) {
        for (size_t i = 0; i < pointNum; ++i) {
            auto& pt  = triPoints[i];
            auto& rgb = triPixels[i];
#ifndef RGBTSDF_USE_BGR_COLOR
            RGBTSDF_FPRINTF(f, "v %f %f %f %d %d %d\n", pt[0], pt[1], pt[2],
                rgb[0], rgb[1], rgb[2]);
#else
            RGBTSDF_FPRINTF(f, "v %f %f %f %d %d %d\n", pt[0], pt[1], pt[2],
                rgb[2], rgb[1], rgb[0]);
#endif
        }
    } else {
        for (size_t i = 0; i < pointNum; ++i) {
            auto& pt = triPoints[i];
            RGBTSDF_FPRINTF(f, "v %f %f %f\n", pt[0], pt[1], pt[2]);
        }
    }
    if (triNormals != nullptr) {
        for (size_t i = 0; i < pointNum; ++i) {
            auto& nl = triNormals[i];
            RGBTSDF_FPRINTF(f, "vn %f %f %f\n", nl[0], nl[1], nl[2]);
        }
        for (size_t i = 0; i < indexNum; ++i) {
            auto triInd = *(const Point3_<_Index>*)triIndexs[i] + 1;
            RGBTSDF_FPRINTF(f, "f %d//%d %d//%d %d//%d\n", triInd.x, triInd.x,
                triInd.y, triInd.y, triInd.z, triInd.z);
        }
    } else {
        for (size_t i = 0; i < indexNum; ++i) {
            auto triInd = *(const Point3_<_Index>*)triIndexs[i] + 1;
            RGBTSDF_FPRINTF(f, "f %d %d %d\n", triInd.x, triInd.y, triInd.z);
        }
    }
    RGBTSDF_FFLUSH(f);
    return RGBTSDF_FCLOSE(f) == 0;
}
template <typename _Float, typename _Index>
static inline bool writeTriMeshToOBJ(const char* meshPath,
    const Point3_<_Float>* triPoints, size_t pointNum,
    const Point3_<_Index>* triIndexs, size_t indexNum)
{
    return writeTriMeshToOBJ(meshPath, (const _Float(*)[3])triPoints, nullptr,
        nullptr, pointNum, (const _Index(*)[3])triIndexs, indexNum);
}
template <typename _Float, typename _Index>
static inline bool writeTriMeshToOBJ(const char* meshPath,
    const Point3_<_Float>*                       triPoints,
    const Point3_<coord_trait_t<_Float>>* triNormals, size_t pointNum,
    Point3_<_Index>* triIndexs, size_t indexNum)
{
    return writeTriMeshToOBJ(meshPath, (const _Float(*)[3])triPoints,
        (const _Float(*)[3])triNormals, nullptr, pointNum,
        (const _Index(*)[3])triIndexs, indexNum);
}
template <typename _Float, typename _Index>
static inline bool writeTriMeshToOBJ(const char* meshPath,
    const Point3_<_Float>*                       triPoints,
    const Point3_<coord_trait_t<_Float>>* triNormals, const RGBPixel* triPixels,
    size_t pointNum, Point3_<_Index>* triIndexs, size_t indexNum)
{
    return writeTriMeshToOBJ(meshPath, (const _Float(*)[3])triPoints,
        (const _Float(*)[3])triNormals, (const unsigned char(*)[3])triPixels,
        pointNum, (const _Index(*)[3])triIndexs, indexNum);
}

const int edgeTable[256] = {0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605,
    0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,
    0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a,
    0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3,
    0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3,
    0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c,
    0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa,
    0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9,
    0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55,
    0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
    0x2cf, 0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf,
    0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f,
    0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0,
    0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff, 0x1f6,
    0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9,
    0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,
    0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93,
    0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393,
    0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c,
    0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};

const int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

}  // namespace rgbtsdf
#endif  // _RGBTSDF_RGBTSDF_HPP_