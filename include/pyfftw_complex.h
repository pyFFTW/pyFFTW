
/*
 * Copyright 2013 Knowledge Economy Developments Ltd
 * 
 * Henry Gomersall
 * heng@kedevelopments.co.uk
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* Defines complex types that are bit compatible with C99's complex.h
 * and (crucially) the same type as expected by fftw3.h.
 * Note, don't use this with complex.h. fftw3.h checks to see whether
 * complex.h is included and then uses that to set the interface.
 * Since MSVC doesn't support C99, by using the following types we
 * have a cross platform/compiler solution.
 *
 * */

#ifndef PYFFTW_COMPLEX_H
#define PYFFTW_COMPLEX_H

typedef float cfloat[2];
typedef double cdouble[2];
typedef long double clongdouble[2];

#endif /* Header guard */
