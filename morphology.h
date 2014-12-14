#pragma once
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
* @function Morphology_Operations
* @param
* Operator - 0:Opening, 1:Closing, 2:Gradient, 3:Top Hat, 4:Black Hat
* Element - 0:Rect, 1:Cross, 2:Ellipse
* size = 2*morph_size + 1
*/
void Morphology_Operations(Mat src, Mat &dst, int morph_operator, int morph_elem, int morph_size)
{
	// Since MORPH_X : 2,3,4,5 and 6
	// Operator - 0:Opening, 1:Closing, 2:Gradient, 3:Top Hat, 4:Black Hat
	int operation = morph_operator + 2;

	// Element - 0:Rect, 1:Cross, 2:Ellipse
	// Size = 2*morph_size + 1
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

	/// Apply the specified morphology operation
	morphologyEx(src, dst, operation, element);
}