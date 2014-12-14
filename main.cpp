//
//  main.cpp
//  Hand
//
//  Created by Nuttapon Pattanavijit on 11/24/14.
//  Copyright (c) 2014 nut. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <deque>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <string>
#include "morphology.h"
#include <fstream>
#include "PlamLinePredictionPart2.hpp"

#define pii pair<int,int>
#define vii vector<pair<int,int> >
#define vint vector<int>
#define dii deque<pair<int,int> >

#define INPUT_SIZE 20
#define OUTPUT_SIZE 4

#define OPENING_SIZE 2
#define CLOSING_SIZE 3
#define OPENING 0
#define CLOSING 1

#define OUTPUTPATH "E:/xampp/htdocs/handserver/app/output/"
#define INPUTPATH "E:/xampp/htdocs/handserver/app/storage/uploads/"
#define NNPATH "E:/xampp/htdocs/handserver/nn.xml"

using namespace cv;
using namespace std;

void rotate2D(const cv::Mat & src, cv::Mat & dst, const double degrees);

// Nasty global variable

Mat img, imgb, imgc, imgd, img_out, imgat, img_out2, img_out3, imgat2;
int dilation_size;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
int blur_size = 1, blur_max = 10;


int gsize = 2, gd = 0;
int bri = 150, con = 10;
//int bri = 170, con = 100;
int equ = 0;
int thresholdVal = 0;

int edgeThresh = 1;
int lowThreshold = 10;
int const max_lowThreshold = 100;
int canny_ratio = 3;
int kernel_size = 3;

int minPts = 5;
// int epsDist = 2;
int epsDist = 17;

int minCluster = 3;
int blockSize = 10;
//int meanSubtract = 2;
int meanSubtract = 1;

vector<Mat> top5mat_output;
vector<int> top5mat_label;
vector<Mat> matNN;
vector<int> labelNN;

//End of nasty global variable

CvANN_MLP loadNN() {
	
	CvANN_MLP nnetwork;

	CvFileStorage* storage = cvOpenFileStorage("nn.xml", 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "nnetwork");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	return nnetwork;
}

int dist(int x1, int x2, int y1, int y2) {
	return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
}

vii neighborQuery(Mat &m, int mi, int mj) {
	int i, j;

	int row = m.rows;
	int col = m.cols;

	vii neighbor;

	for (i = mi - epsDist; i <= mi + epsDist; i++) {
		for (j = mj - epsDist; j <= mj + epsDist; j++) {

			if (i >= 0 && i < row && j >= 0 && j < col) {

				if (m.at<unsigned char>(i, j)) {
					if (dist(i, mi, j, mj) <= epsDist * 2) {
						neighbor.push_back(pii(i, j));
					}
				}
			}

		}
	}

	return neighbor;
}

bool mycompare(pii a, pii b) {
	return a.second > b.second;
}

Mat DBScan(Mat input) {

	top5mat_output.clear();
	top5mat_label.clear();

	Mat output_rgb[3];
	output_rgb[0] = Mat::zeros(input.size(), CV_8UC1);
	output_rgb[1] = Mat::zeros(input.size(), CV_8UC1);
	output_rgb[2] = Mat::zeros(input.size(), CV_8UC1);
	Mat output;
	Mat cluster = Mat::zeros(input.size(), CV_32SC1);
	Mat visit = Mat::zeros(input.size(), CV_8UC1);
	Mat noise = Mat::zeros(input.size(), CV_8UC1);


	int row = input.rows;
	int col = input.cols;

	int i, j, k, l;

	int clusterNo = 0;

	dii queue;
	vii count;
	count.push_back(pii(0, 0));

	for (i = 0; i<row; i++) {
		for (j = 0; j<col; j++) {

			queue.clear();

			if (!visit.at<unsigned char>(i, j) && input.at<unsigned char>(i, j)) {

				visit.at<unsigned char>(i, j) = 255;

				vii neighbor = neighborQuery(input, i, j);

				if (neighbor.size() < minPts) {
					noise.at<unsigned char>(i, j) = 255;
					//cout << "noise" << rand() << endl;
				}
				else {
					clusterNo++;
					count.push_back(pii(clusterNo, 0));
					int thisCluster = clusterNo;


					for (k = 0; k<neighbor.size(); k++) {
						queue.push_back(neighbor[k]);
					}
					cluster.at<int>(i, j) = thisCluster;

					// Expand cluster

					while (!queue.empty()) {
						pii p = queue.front();
						queue.pop_front();

						if (!visit.at<unsigned char>(p.first, p.second)) {
							visit.at<unsigned char>(p.first, p.second) = 255;
							neighbor = neighborQuery(input, p.first, p.second);

							if (neighbor.size() >= minPts) {
								for (k = 0; k<neighbor.size(); k++) {
									queue.push_back(neighbor[k]);
								}
							}
						}
						if (cluster.at<int>(p.first, p.second) == 0) {
							cluster.at<int>(p.first, p.second) = thisCluster;
							count[thisCluster].second++;
						}


					}


				}
			}
		}
	}

	sort(count.begin(), count.end(), mycompare);

	vint top5;

	for (i = 0; i<5; i++) {
		top5.push_back(count[i].first);
	}

	int c;

	//    clusterX.clear();
	//    clusterY.clear();

	for (int i = 0; i<5; i++) {
		vector<double> vecx;
		vector<double> vecy;
		//        clusterX.push_back(vecx);
		//        clusterY.push_back(vecy);
	}

	Mat top5mat[5];
	for (i = 0; i<5; i++) {
		top5mat[i] = Mat::zeros(input.rows, input.cols, CV_8UC1);
	}

	//cout << "--------- Start Data ----------" << endl;
	for (i = 0; i<row; i++) {
		for (j = 0; j<col; j++) {
			c = cluster.at<int>(i, j);
			if (c && find(top5.begin(), top5.end(), c) != top5.end()) {
				srand(c);
				output_rgb[0].at<unsigned char>(i, j) = rand() % 156 + 100;
				output_rgb[1].at<unsigned char>(i, j) = rand() % 156 + 100;
				output_rgb[2].at<unsigned char>(i, j) = rand() % 156 + 100;


				int pos = (int)(find(top5.begin(), top5.end(), c) - top5.begin() + 1);

				// cout << pos << " " << i << " " << j << endl;

				//                clusterX[pos-1].push_back(i);
				//                clusterY[pos-1].push_back(j);

				top5mat[pos - 1].at<unsigned char>(i, j) = 255;

			}
			else if (c) {
				srand(c);
				/*
				output_rgb[0].at<unsigned char>(i,j) = rand() % 50;
				output_rgb[1].at<unsigned char>(i,j) = rand() % 50;
				output_rgb[2].at<unsigned char>(i,j) = rand() % 50;
				*/
				output_rgb[0].at<unsigned char>(i, j) = 0;
				output_rgb[1].at<unsigned char>(i, j) = 0;
				output_rgb[2].at<unsigned char>(i, j) = 0;

			}
		}
	}

	//cout << "--------- End Data ----------" << endl;

	merge(output_rgb, 3, output);

	//    imshow("cluster", output);

	for (i = 0; i<5; i++) {
		top5mat_output.push_back(top5mat[i]);
		//        imwrite(outpath+filename+"_"+to_string(i)+filetypeout, top5mat[i]);
	}

	return output;
}

int main(int argc, const char * argv[]) {

	// insert code here...

	namedWindow("test");
	string filename = argv[1];

#ifdef _DEBUG
	string outputPath = "c:/ai/";
	string inputPath = "C:/ai/ok.jpg";
#else
	string outputPath = OUTPUTPATH;
	string inputPath = INPUTPATH + filename + ".jpg";
#endif

	string outputImgPath = outputPath + filename + ".jpg";
	string outputTxtPath = outputPath + filename + ".txt";

	//int ibegin = 975;
	//int iend = 1845;
	//int jbegin = 355;
	//int jend = 1225;
	int ibegin = 434;
	int iend = 834;
	int jbegin = 185;
	int jend = 585;

	Mat img_input = imread(inputPath);

	resize(img_input, img_input, Size(866, 650));

	rotate2D(img_input, img_input, 270.0);

	//Mat test_img;
	//img_input.copyTo(test_img);
	//rectangle(test_img, Point(ibegin, jbegin), Point(iend, jend), Scalar(255, 0, 0));
	//imshow("input_rect", test_img);

	imshow("original_input", img_input);
	Mat img = Mat::zeros(iend-ibegin, jend-jbegin, CV_8UC3);

	for (int i = ibegin; i<iend; i++) {
		for (int j = jbegin; j<jend; j++) {
			Vec3b a = img_input.at<Vec3b>(i, j);
			img.at<Vec3b>(i - ibegin, j - jbegin) = a;
		}
	}
	printf("rotate&crop is finished\n");
	resize(img, img, Size(400, 400));
	Mat original_img;
	img.copyTo(original_img);
	//imshow("test", img_input);
	//waitKey(1000);

	//imshow("test", img);
	//waitKey(1000);

	cvtColor(img, img, CV_BGR2GRAY);

	GaussianBlur(img, imgc, Size(gsize * 2 + 1, gsize * 2 + 1), gd);

	//imshow("test", imgc);
	//waitKey(1000);

	imgc.convertTo(imgd, -1, 1 + con / 100.0, bri / 10.0 - 15);

	imshow("test", imgd);
	waitKey(500);

	adaptiveThreshold(imgd, imgat, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize * 2 + 1, meanSubtract);
	imshow("adaptiveThreshold", imgat);
	waitKey(500);
	printf("adaptiveThreshold is finished\n");

	imgat2 = DBScan(255 - imgat);
	imshow("DBScan", imgat2);
	waitKey(500);
	printf("DBScan is finished\n");

	// Neural Network

	Mat test1, test2, test1out;

	CvANN_MLP nnetwork;// = loadNN();
	CvFileStorage* storage = cvOpenFileStorage(NNPATH, 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "nnetwork");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	printf("loadNN finish\n");
	printf("go to loop\n");
	for (int i = 0; i < top5mat_output.size(); i++) {
		top5mat_output[i].copyTo(test1);
		top5mat_output[i].copyTo(test2);

		//        imshow("test", test2);
		//        waitKey(1000);

		test1.convertTo(test1, CV_32F);
		resize(test1, test1, Size(INPUT_SIZE, INPUT_SIZE));
		test1 = test1.reshape(0, 1);

		nnetwork.predict(test1, test1out);

		float max = -9999999.0;
		int max_id = -1;
		for (int j = 0; j<test1out.cols; j++) {
			float val = test1out.at<float>(0, j);
			//            cout << j << ":" << val << endl;
			if (val > max) {
				max = val;
				max_id = j;
			}
		}

		Mat test3[5];

		test2.copyTo(test3[i]);

		if (max > 0.2) {
			//            imshow("test", test3[i]);
			//            waitKey(1000);
			matNN.push_back(test3[i]);
			labelNN.push_back(max_id);
		}
	}
	printf("clustering is finished\n");

	Mat img_fin;

	original_img.copyTo(img_fin);

	for (int k = 0; k < matNN.size(); k++) {
		Mat line;
		matNN[k].copyTo(line);

		//string name = "line" + std::to_string(k);
		//imshow(name, line);
		Morphology_Operations(line, line, CLOSING, 2, CLOSING_SIZE);
		//imshow(name + "(1)", line);
		Morphology_Operations(line, line, OPENING, 2, OPENING_SIZE);
		//imshow(name+"(2)", line);

		for (int i = 0; i < line.rows; i++) {
			for (int j = 0; j < line.cols; j++) {
				if (line.at<unsigned char>(i, j)) {
					//img_fin.at<Vec3b>(i + ibegin, j + jbegin) = Vec3b(255, 255, 255);
					img_fin.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

	}
	printf("draw line is finished\n");

	// Ask Oracle
	string oracle = predict(matNN, labelNN);
	std::ofstream outfile;
	outfile.open(outputTxtPath);
	outfile << oracle;
	outfile.close();


	imwrite(outputImgPath, img_fin);
	imshow("output", img_fin);

#ifdef _DEBUG
	waitKey(0);
#endif

	return 0;
}

void rotate2D(const cv::Mat & src, cv::Mat & dst, const double degrees)
{
	cv::Mat frame, frameRotated;

	int diagonal = (int)sqrt(src.cols * src.cols + src.rows * src.rows);
	int newWidth = diagonal;
	int newHeight = diagonal;

	int offsetX = (newWidth - src.cols) / 2;
	int offsetY = (newHeight - src.rows) / 2;
	cv::Mat targetMat(newWidth, newHeight, src.type(), cv::Scalar(0));
	cv::Point2f src_center(targetMat.cols / 2.0f, targetMat.rows / 2.0f);

	src.copyTo(frame);

	frame.copyTo(targetMat.rowRange(offsetY, offsetY +
		frame.rows).colRange(offsetX, offsetX + frame.cols));
	cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, degrees, 1.0);
	cv::warpAffine(targetMat, frameRotated, rot_mat, targetMat.size());

	cv::Rect bound_Rect(frame.cols, frame.rows, 0, 0);
	int x1 = offsetX;
	int x2 = offsetX + frame.cols;
	int x3 = offsetX;
	int x4 = offsetX + frame.cols;
	int y1 = offsetY;
	int y2 = offsetY;
	int y3 = offsetY + frame.rows;
	int y4 = offsetY + frame.rows;
	cv::Mat co_Ordinate = (cv::Mat_<double>(3, 4) << x1, x2, x3, x4,
		y1, y2, y3, y4,
		1, 1, 1, 1);

	cv::Mat RotCo_Ordinate = rot_mat * co_Ordinate;

	for (int i = 0; i < 4; ++i) {
		if (RotCo_Ordinate.at<double>(0, i) < bound_Rect.x)
			bound_Rect.x = (int)RotCo_Ordinate.at<double>(0, i);
		if (RotCo_Ordinate.at<double>(1, i) < bound_Rect.y)
			bound_Rect.y = RotCo_Ordinate.at<double>(1, i);
	}

	for (int i = 0; i < 4; ++i) {
		if (RotCo_Ordinate.at<double>(0, i) > bound_Rect.width)
			bound_Rect.width = (int)RotCo_Ordinate.at<double>(0, i);
		if (RotCo_Ordinate.at<double>(1, i) > bound_Rect.height)
			bound_Rect.height = RotCo_Ordinate.at<double>(1, i);
	}

	bound_Rect.width = bound_Rect.width - bound_Rect.x;
	bound_Rect.height = bound_Rect.height - bound_Rect.y;

	if (bound_Rect.x < 0)
		bound_Rect.x = 0;
	if (bound_Rect.y < 0)
		bound_Rect.y = 0;
	if (bound_Rect.width > frameRotated.cols)
		bound_Rect.width = frameRotated.cols;
	if (bound_Rect.height > frameRotated.rows)
		bound_Rect.height = frameRotated.rows;

	cv::Mat ROI = frameRotated(bound_Rect);
	ROI.copyTo(dst);

}
