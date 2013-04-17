/*
 * simple_arrays.h
 *
 *  Created on: Apr 10, 2013
 *      Author: hasegawa
 */

#ifndef SIMPLE_ARRAYS_H_
#define SIMPLE_ARRAYS_H_

#include <stdint.h>


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
    }/*
    Image8U(PyObject *obj) : nd(0), dims(0), size(0), raster(0)
    {
        PyArrayObject *f = (PyArrayObject *)obj;
        if((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_UBYTE) && PyArray_ISCARRAY(f)) {
            set_dims(f->nd, (int *)f->dimensions);
            raster = (char *)new unsigned char[size];
            memcpy(raster, f->data, size*sizeof(unsigned char));
        }
    }*/
    ~Image8U() {
        delete dims;
        delete raster;
    }/*
    PyObject *get_numpy() const
    {
        PyArrayObject *obj = (PyArrayObject *)PyArray_SimpleNew(nd, (npy_intp *)dims, NPY_UBYTE);
        memcpy(obj->data, raster, size*sizeof(unsigned char));
        return (PyObject *)obj;
    }*/
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
//    Image32F Class
// ================================================================================

class Image32F {
public:
	int nd;
	int size;
	int *dims;
	char *raster;
	Image32F() :
			nd(0), dims(0), size(0), raster(0) {
	}
	Image32F(int _nd, int *_dims, float *_raster) :
			nd(0), dims(0), size(0), raster(0) {
		set_dims(_nd, _dims);
		raster = (char *) new float[size];
		memcpy(raster, _raster, size * sizeof(float));
	}
	Image32F(int _nd, int *_dims) :
			nd(0), dims(0), size(0), raster(0) {
		set_dims(_nd, _dims);
		raster = (char *) new float[size];
		memset(raster, 0, size * sizeof(float));
	}
	Image32F(Image32F *f) :
			nd(0), dims(0), size(0), raster(0) {
		if ((f) && (f->raster)) {
			set_dims(f->nd, f->dims);
			raster = (char *) new float[size];
			memcpy(raster, f->raster, size * sizeof(float));
		}
	}
	/*
	Image32F(PyObject *obj) :
			nd(0), dims(0), size(0), raster(0) {
		PyArrayObject *f = (PyArrayObject *) obj;
		if ((f) && PyArray_Check(f) && (PyArray_TYPE(f) == NPY_FLOAT)
				&& PyArray_ISCARRAY(f)) {
			set_dims(f->nd, (int *) f->dimensions);
			raster = (char *) new float[size];
			memcpy(raster, f->data, size * sizeof(float));
		}
	}*/
	~Image32F() {
		delete dims;
		delete raster;
	}
	/*
	PyObject *get_numpy() const {
		PyArrayObject *obj = (PyArrayObject *) PyArray_SimpleNew(nd,
				(npy_intp *) dims, NPY_FLOAT);
		memcpy(obj->data, raster, size * sizeof(float));
		return (PyObject *) obj;
	}*/
	void copy(Image32F *f) {
		if ((f) && (f->raster)) {
			set_dims(f->nd, f->dims);
			raster = (char *) new float[size];
			memcpy(raster, f->raster, size * sizeof(float));
		}
	}
	void set_dims(int _nd, int *_dims) {
		nd = _nd;
		size = 1;
		dims = new int[nd];
		for (int i = 0; i < nd; i++) {
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
	cv::Mat _mat;
	Image32F(cv::Mat mat) : nd(0), dims(0), size(0), raster(0)
	{
		if(mat.type() != CV_32F) return;
		_mat = mat;
		nd = 2;
		dims = new int[2];
		dims[0] = mat.rows;
		dims[1] = mat.cols;
		size = 0[dims] * 1[dims];
	}
	IplImage *get_opencv() const
	{
		return (IplImage *)0;
	}
#endif
};

#endif /* SIMPLE_ARRAYS_H_ */
