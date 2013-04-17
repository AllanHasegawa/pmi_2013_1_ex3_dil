/*
 * dil.h
 *
 *  Created on: Apr 11, 2013
 *      Author: hasegawa
 */

#ifndef DIL_H_
#define DIL_H_

class Image32F;

Image32F* cudaBinaryDilation(const Image32F& h_img, const Image32F& h_se);

#endif /* DIL_H_ */
