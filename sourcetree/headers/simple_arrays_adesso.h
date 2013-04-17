/*
 * simple_arrays_adesso.h
 *
 *  Created on: Apr 10, 2013
 *      Author: hasegawa
 */

#ifndef SIMPLE_ARRAYS_ADESSO_H_
#define SIMPLE_ARRAYS_ADESSO_H_



#include "Python.h"
    #include "numpy/arrayobject.h"
    #ifdef REQUIRE_MORPH
    #include "morph4python.h"
    #endif
    #ifdef REQUIRE_OPENCV
    #include "cxcore.h"
    #endif

    #define pyprintf PySys_WriteStdout

    // ================================================================================
    //    Image8U Class
    // ================================================================================

    class Image8U {
    public:
        int nd;
        int size;
        int *dims;
        char *raster;
        Image8U() : nd(0), dims(0), size(0), raster(0)
        {}
        Image8U(int _nd, int *_dims, unsigned char *_raster) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new unsigned char[size];
            memcpy(raster, _raster, size*sizeof(unsigned char));
        }
        Image8U(int _nd, int *_dims) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new unsigned char[size];
            memset(raster, 0, size*sizeof(unsigned char));
        }
        Image8U(Image8U *f) : nd(0), dims(0), size(0), raster(0)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new unsigned char[size];
                memcpy(raster, f->raster, size*sizeof(unsigned char));
            }
        }
        Image8U(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
        {
            PyArrayObject *f = (PyArrayObject *)obj;
            if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_UBYTE) && PyArray_ISCARRAY(f)) {
                set_dims(f->nd, (int *)f->dimensions);
                raster = (char *)new unsigned char[size];
                memcpy(raster, f->data, size*sizeof(unsigned char));
            }
        }
        ~Image8U() {
            delete dims;
            delete raster;
        }
        PyObject *get_numpy() const
        {
            PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_UBYTE);
            memcpy(obj->data, raster, size*sizeof(unsigned char));
            return (PyObject *)obj;
        }
        void copy(Image8U *f)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new unsigned char[size];
                memcpy(raster, f->raster, size*sizeof(unsigned char));
            }
        }
        void set_dims(int _nd, int *_dims)
        {
            nd = _nd;
            size = 1;
            dims = new int[nd];
            for(int i = 0; i < nd; i++) {
                dims[i] = _dims[i];
                size *= dims[i];
            }
        }
    #ifdef REQUIRE_MORPH
        Image8U(Image *img) : nd(0), dims(0), size(0), raster(0)
        {
            if(img->typecode() != MM_UBYTE) return;
            nd = 3;
            dims = new int[3];
            dims[0] = img->depth();
            dims[1] = img->height();
            dims[2] = img->width();
            size = dims[0] * dims[1] * dims[2];
            raster = (char *)new unsigned char[size];
            memcpy(raster, img->raster(), size*sizeof(unsigned char));
        }
        Image *get_morph() const
        {
            if(raster == 0) return 0;
            return new Image(dims[2], dims[1], dims[0], "uint8", raster);
        }
    #endif
    #ifdef REQUIRE_OPENCV
        Image8U(IplImage *ipl) : nd(0), dims(0), size(0), raster(0)
        {
            if(ipl->depth != IPL_DEPTH_8U) return;
        }
        IplImage *get_opencv() const
        {
            return (IplImage *)0;
        }
    #endif
    };

    // ================================================================================
    //    Image16U Class
    // ================================================================================

    class Image16U {
    public:
        int nd;
        int size;
        int *dims;
        char *raster;
        Image16U() : nd(0), dims(0), size(0), raster(0)
        {}
        Image16U(int _nd, int *_dims, unsigned short *_raster) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new unsigned short[size];
            memcpy(raster, _raster, size*sizeof(unsigned short));
        }
        Image16U(int _nd, int *_dims) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new unsigned short[size];
            memset(raster, 0, size*sizeof(unsigned short));
        }
        Image16U(Image16U *f) : nd(0), dims(0), size(0), raster(0)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new unsigned short[size];
                memcpy(raster, f->raster, size*sizeof(unsigned short));
            }
        }
        Image16U(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
        {
            PyArrayObject *f = (PyArrayObject *)obj;
            if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_USHORT) && PyArray_ISCARRAY(f)) {
                set_dims(f->nd, (int *)f->dimensions);
                raster = (char *)new unsigned short[size];
                memcpy(raster, f->data, size*sizeof(unsigned short));
            }
        }
        ~Image16U() {
            delete dims;
            delete raster;
        }
        PyObject *get_numpy() const
        {
            PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_USHORT);
            memcpy(obj->data, raster, size*sizeof(unsigned short));
            return (PyObject *)obj;
        }
        void copy(Image16U *f)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new unsigned short[size];
                memcpy(raster, f->raster, size*sizeof(unsigned short));
            }
        }
        void set_dims(int _nd, int *_dims)
        {
            nd = _nd;
            size = 1;
            dims = new int[nd];
            for(int i = 0; i < nd; i++) {
                dims[i] = _dims[i];
                size *= dims[i];
            }
        }
    #ifdef REQUIRE_MORPH
        Image16U(Image *img) : nd(0), dims(0), size(0), raster(0)
        {
            if(img->typecode() != MM_USHORT) return;
            nd = 3;
            dims = new int[3];
            dims[0] = img->depth();
            dims[1] = img->height();
            dims[2] = img->width();
            size = dims[0] * dims[1] * dims[2];
            raster = (char *)new unsigned short[size];
            memcpy(raster, img->raster(), size*sizeof(unsigned short));
        }
        Image *get_morph() const
        {
            if(raster == 0) return 0;
            return new Image(dims[2], dims[1], dims[0], "uint16", raster);
        }
    #endif
    #ifdef REQUIRE_OPENCV
        Image16U(IplImage *ipl) : nd(0), dims(0), size(0), raster(0)
        {
            if(ipl->depth != IPL_DEPTH_16U) return;
        }
        IplImage *get_opencv() const
        {
            return (IplImage *)0;
        }
    #endif
    };

    // ================================================================================
    //    Image32S Class
    // ================================================================================

    class Image32S {
    public:
        int nd;
        int size;
        int *dims;
        char *raster;
        Image32S() : nd(0), dims(0), size(0), raster(0)
        {}
        Image32S(int _nd, int *_dims, int *_raster) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new int[size];
            memcpy(raster, _raster, size*sizeof(int));
        }
        Image32S(int _nd, int *_dims) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new int[size];
            memset(raster, 0, size*sizeof(int));
        }
        Image32S(Image32S *f) : nd(0), dims(0), size(0), raster(0)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new int[size];
                memcpy(raster, f->raster, size*sizeof(int));
            }
        }
        Image32S(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
        {
            PyArrayObject *f = (PyArrayObject *)obj;
            if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_INT) && PyArray_ISCARRAY(f)) {
                set_dims(f->nd, (int *)f->dimensions);
                raster = (char *)new int[size];
                memcpy(raster, f->data, size*sizeof(int));
            }
        }
        ~Image32S() {
            delete dims;
            delete raster;
        }
        PyObject *get_numpy() const
        {
            PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_INT);
            memcpy(obj->data, raster, size*sizeof(int));
            return (PyObject *)obj;
        }
        void copy(Image32S *f)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new int[size];
                memcpy(raster, f->raster, size*sizeof(int));
            }
        }
        void set_dims(int _nd, int *_dims)
        {
            nd = _nd;
            size = 1;
            dims = new int[nd];
            for(int i = 0; i < nd; i++) {
                dims[i] = _dims[i];
                size *= dims[i];
            }
        }
    #ifdef REQUIRE_MORPH
        Image32S(Image *img) : nd(0), dims(0), size(0), raster(0)
        {
            if(img->typecode() != MM_INT) return;
            nd = 3;
            dims = new int[3];
            dims[0] = img->depth();
            dims[1] = img->height();
            dims[2] = img->width();
            size = dims[0] * dims[1] * dims[2];
            raster = (char *)new int[size];
            memcpy(raster, img->raster(), size*sizeof(int));
        }
        Image *get_morph() const
        {
            if(raster == 0) return 0;
            return new Image(dims[2], dims[1], dims[0], "int32", raster);
        }
    #endif
    #ifdef REQUIRE_OPENCV
        Image32S(IplImage *ipl) : nd(0), dims(0), size(0), raster(0)
        {
            if(ipl->depth != IPL_DEPTH_32S) return;
        }
        IplImage *get_opencv() const
        {
            return (IplImage *)0;
        }
    #endif
    };

    // ================================================================================
    //    Image32F Class
    // ================================================================================

    class Image32F {
    public:
        int nd;
        int size;
        int *dims;
        char *raster;
        Image32F() : nd(0), dims(0), size(0), raster(0)
        {}
        Image32F(int _nd, int *_dims, float *_raster) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new float[size];
            memcpy(raster, _raster, size*sizeof(float));
        }
        Image32F(int _nd, int *_dims) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new float[size];
            memset(raster, 0, size*sizeof(float));
        }
        Image32F(Image32F *f) : nd(0), dims(0), size(0), raster(0)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new float[size];
                memcpy(raster, f->raster, size*sizeof(float));
            }
        }
        Image32F(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
        {
            PyArrayObject *f = (PyArrayObject *)obj;
            if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_FLOAT) && PyArray_ISCARRAY(f)) {
                set_dims(f->nd, (int *)f->dimensions);
                raster = (char *)new float[size];
                memcpy(raster, f->data, size*sizeof(float));
            }
        }
        ~Image32F() {
            delete dims;
            delete raster;
        }
        PyObject *get_numpy() const
        {
            PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_FLOAT);
            memcpy(obj->data, raster, size*sizeof(float));
            return (PyObject *)obj;
        }
        void copy(Image32F *f)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new float[size];
                memcpy(raster, f->raster, size*sizeof(float));
            }
        }
        void set_dims(int _nd, int *_dims)
        {
            nd = _nd;
            size = 1;
            dims = new int[nd];
            for(int i = 0; i < nd; i++) {
                dims[i] = _dims[i];
                size *= dims[i];
            }
        }
    #ifdef REQUIRE_MORPH
        Image32F(Image *img) : nd(0), dims(0), size(0), raster(0)
        {
            if(img->typecode() != MM_FLOAT) return;
            nd = 3;
            dims = new int[3];
            dims[0] = img->depth();
            dims[1] = img->height();
            dims[2] = img->width();
            size = dims[0] * dims[1] * dims[2];
            raster = (char *)new float[size];
            memcpy(raster, img->raster(), size*sizeof(float));
        }
        Image *get_morph() const
        {
            if(raster == 0) return 0;
            return new Image(dims[2], dims[1], dims[0], "float32", raster);
        }
    #endif
    #ifdef REQUIRE_OPENCV
        Image32F(IplImage *ipl) : nd(0), dims(0), size(0), raster(0)
        {
            if(ipl->depth != IPL_DEPTH_32F) return;
        }
        IplImage *get_opencv() const
        {
            return (IplImage *)0;
        }
    #endif
    };

    // ================================================================================
    //    Image64F Class
    // ================================================================================

    class Image64F {
    public:
        int nd;
        int size;
        int *dims;
        char *raster;
        Image64F() : nd(0), dims(0), size(0), raster(0)
        {}
        Image64F(int _nd, int *_dims, double *_raster) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new double[size];
            memcpy(raster, _raster, size*sizeof(double));
        }
        Image64F(int _nd, int *_dims) : nd(0), dims(0), size(0), raster(0)
        {
            set_dims(_nd, _dims);
            raster = (char *)new double[size];
            memset(raster, 0, size*sizeof(double));
        }
        Image64F(Image64F *f) : nd(0), dims(0), size(0), raster(0)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new double[size];
                memcpy(raster, f->raster, size*sizeof(double));
            }
        }
        Image64F(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
        {
            PyArrayObject *f = (PyArrayObject *)obj;
            if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_DOUBLE) && PyArray_ISCARRAY(f)) {
                set_dims(f->nd, (int *)f->dimensions);
                raster = (char *)new double[size];
                memcpy(raster, f->data, size*sizeof(double));
            }
        }
        ~Image64F() {
            delete dims;
            delete raster;
        }
        PyObject *get_numpy() const
        {
            PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_DOUBLE);
            memcpy(obj->data, raster, size*sizeof(double));
            return (PyObject *)obj;
        }
        void copy(Image64F *f)
        {
            if((f) && (f->raster)) {
                set_dims(f->nd, f->dims);
                raster = (char *)new double[size];
                memcpy(raster, f->raster, size*sizeof(double));
            }
        }
        void set_dims(int _nd, int *_dims)
        {
            nd = _nd;
            size = 1;
            dims = new int[nd];
            for(int i = 0; i < nd; i++) {
                dims[i] = _dims[i];
                size *= dims[i];
            }
        }
    #ifdef REQUIRE_MORPH
        Image64F(Image *img) : nd(0), dims(0), size(0), raster(0)
        {
            if(img->typecode() != MM_DOUBLE) return;
            nd = 3;
            dims = new int[3];
            dims[0] = img->depth();
            dims[1] = img->height();
            dims[2] = img->width();
            size = dims[0] * dims[1] * dims[2];
            raster = (char *)new double[size];
            memcpy(raster, img->raster(), size*sizeof(double));
        }
        Image *get_morph() const
        {
            if(raster == 0) return 0;
            return new Image(dims[2], dims[1], dims[0], "float64", raster);
        }
    #endif
    #ifdef REQUIRE_OPENCV
        Image64F(IplImage *ipl) : nd(0), dims(0), size(0), raster(0)
        {
            if(ipl->depth != IPL_DEPTH_64F) return;
        }
        IplImage *get_opencv() const
        {
            return (IplImage *)0;
        }
    #endif
    };

#endif /* SIMPLE_ARRAYS_ADESSO_H_ */
