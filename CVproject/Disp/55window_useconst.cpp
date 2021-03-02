#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

#define CONST 60
#define OCCLUSIONCOST 400
enum { DONE, DIAG, UP, LEFT };

struct cost_case {
	double cost;
	int move;
	double dsi;
};

void onMouseEvent(int event, int x, int y, int flags, void* dstImage) {
	Mat mouseImage = *(Mat*)dstImage;
	switch (event) {
		// ���콺 ���� Ŭ�� �̺�Ʈ.. 
	case EVENT_LBUTTONDOWN:
		cout << Point(x, y) << endl; break;
	}
}

int main()
{
	// left image
	Mat image1 = imread("im2.png", IMREAD_COLOR);
	if (image1.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	// right image
	Mat image2 = imread("im6.png", IMREAD_COLOR);
	if (image2.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	// resize(image1, image1, Size(150, 100), 0, 0, INTER_LINEAR);
	// resize(image2, image2, Size(150, 100), 0, 0, INTER_LINEAR);

	int width = image1.cols;
	int height = image1.rows;

	Mat pad_image1(height + 4, width + 4, CV_8UC3); // pading
	Mat pad_image2(height + 4, width + 4, CV_8UC3); // pading
	Mat simpoint_image(height, width, CV_8UC1); // similar point image
	Mat disp_image(width, width, CV_8UC1); // DSI image ����
	Mat path(width, width, CV_8UC1);

	simpoint_image.setTo(0);

	// cost�� �̵� case�� ������ matrix ���� width *
	cost_case** matrix = new cost_case * [width + 1];
	for (int i = 0; i < width + 1; i++)
		matrix[i] = new cost_case[width + 1];

	double** error_matrix = new double * [width];
	for (int i = 0; i < width; i++)
		error_matrix[i] = new double[width];

	double* errorbox = new double[width];

	// matrix �ʱ� ����
	for (int i = 0; i < width + 1; i++) {
		for (int j = 0; j < width + 1; j++) {
			if (i == 0) { // ù�� cost, move ����
				matrix[i][j].cost = OCCLUSIONCOST * j;
				if (j == 0)
					matrix[i][j].move = DONE;
				else
					matrix[i][j].move = LEFT;
			}
			else {
				if (j == 0) { // ù�� cost, move ����
					matrix[i][j].cost = OCCLUSIONCOST * i;
					matrix[i][j].move = UP; // UP -> DIAG
				}
			}
		}
	}

	// padding
	copyMakeBorder(image1, pad_image1, 2, 2, 2, 2, BORDER_REPLICATE);
	copyMakeBorder(image2, pad_image2, 2, 2, 2, 2, BORDER_REPLICATE);

	// DSI ���ϱ�
	for (int y = 2; y < height + 2; y++) {
	//for (int y = 70; y < 110; y++) {
	//int y = 190;
		disp_image.setTo(0);
		// y�� DSI�� ���ϰ� DP
		for (int x = 2; x < width + 2; x++) {
			// �ּҸ� �����Ϳ� ����
			uchar* pointer_output = disp_image.ptr<uchar>(x - 2);

			// row �����ͷκ��� (x * 3)��° ������ �ִ� �ȼ��� �����ɴϴ�.
			//0, 1, 2 ������� blue, green, red ä�ΰ��� ������ ���ִ� ������ �ϳ��� �ȼ��� �޸𸮻� b g r ������� ����Ǳ� �����Դϴ�. 
			double max = 0, error;
			for (int k = 2; k < width + 2; k++) {
				error = 0;
				int l = k - 2;
				if (x < CONST) {
					if (l < x) {
						for (int winh = -2; winh <= 2; winh++) { // 5 * 5 window dis
							for (int winv = -2; winv <= 2; winv++) {
								uchar b1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 0];
								uchar g1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 1];
								uchar r1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 2];
								uchar b2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 0];
								uchar g2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 1];
								uchar r2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 2];
								error += (pow(b1 - b2, 2) + pow(g1 - g2, 2) + pow(r1 - r2, 2));
							}
						}
						errorbox[l] = error;
						matrix[x - 1][l + 1].dsi = error / 50;
						max = (max < error) ? error : max;
					}
					else {
						pointer_output[l] = 255;
						matrix[x - 1][l + 1].dsi = 100000;
					}
				}
				else {
					if (l < x - CONST) {
						pointer_output[l] = 255;
						matrix[x - 1][l + 1].dsi = 100000;
					}
					else if ((x - CONST < l) && (l < x - 1)) {
						for (int winh = -2; winh <= 2; winh++) { // 5 * 5 window dis
							for (int winv = -2; winv <= 2; winv++) {
								uchar b1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 0];
								uchar g1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 1];
								uchar r1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 2];
								uchar b2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 0];
								uchar g2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 1];
								uchar r2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 2];
								error += (pow(b1 - b2, 2) + pow(g1 - g2, 2) + pow(r1 - r2, 2));
							}
						}
						errorbox[l] = error;
						matrix[x - 1][l + 1].dsi = error / 50;
						max = (max < error) ? error : max;
					}
					else {
						pointer_output[l] = 255;
						matrix[x - 1][l + 1].dsi = 100000;
					}
				}
			}

			for (int k = 0; k < width; k++) {
				if (pointer_output[k] != 255)
					pointer_output[k] = errorbox[k] / max * 255;
			}

			if (x > 2) // DP�� ���� cost�� move ����
				for (int m = 1; m < width + 1; m++) {
					int min1 = matrix[x - 3][m - 1].cost + matrix[x - 2][m].dsi;
					int min2 = matrix[x - 3][m].cost + OCCLUSIONCOST;
					int min3 = matrix[x - 2][m - 1].cost + OCCLUSIONCOST;

					int lastmin = min(min1, min2);
					lastmin = min(min3, lastmin);

					matrix[x - 2][m].cost = lastmin;
					if (min1 == lastmin) matrix[x - 2][m].move = DIAG;
					if (min3 == lastmin) matrix[x - 2][m].move = LEFT;
					if (min2 == lastmin) matrix[x - 2][m].move = UP;
				}
		}
		for (int m = 1; m < width + 1; m++) {
			int min1 = matrix[width - 1][m - 1].cost + matrix[width - 1][m].dsi;
			int min2 = matrix[width][m].cost + OCCLUSIONCOST;
			int min3 = matrix[width][m - 1].cost + OCCLUSIONCOST;

			int lastmin = min(min1, min2);
			lastmin = min(min3, lastmin);

			matrix[width][m].cost = lastmin;
			if (min1 == lastmin) matrix[width][m].move = DIAG;
			if (min2 == lastmin) matrix[width][m].move = UP;
			if (min3 == lastmin) matrix[width][m].move = LEFT;
		}
		/*
		ofstream file("matrix_cost.txt");
		for (int m = 0; m < width + 1; m++) {
			for (int k = 0; k < width + 1; k++) {
				file << matrix[m][k].cost << ' ';
			}
			file << endl;
		}
		file.close();
		ofstream file2("matrix_move.txt");
		for (int m = 0; m < width + 1; m++) {
			for (int k = 0; k < width + 1; k++) {
				file2 << matrix[m][k].move << ' ';
			}
			file2 << endl;
		}
		file2.close();
		ofstream file3("matrix_dsi.txt");
		for (int m = 0; m < width + 1; m++) {
			for (int k = 0; k < width + 1; k++) {
				file3 << matrix[m][k].dsi << ' ';
			}
			file3 << endl;
		}
		file3.close();
		*/
		uchar* pointer_simimage = simpoint_image.ptr<uchar>(y - 2); // sim image y point
		// case1 : disparity ��� x
		// path ���� ���ϱ�
		int p = width - 1, q = width - 1;
		int point = p;
		while (p != 0 && q != 0) {
			switch (matrix[p][q].move) {
			case DIAG:
				pointer_simimage[point] = abs(point - q) * 256 / 60;
				p--;
				q--;
				point--;
				break;
			case UP:
				pointer_simimage[point] = 0;
				p--;
				point--;
				break;
			case LEFT:
				q--;
				break;
			}
			path.at<uchar>(p, q) = 0;
		}
		cout << "now y = " << y << endl;
		//imshow("Disparity", disp_image);
		//waitKey(0);
	}

	namedWindow("Disparity", WINDOW_AUTOSIZE);
	imshow("left Original", pad_image1);
	imshow("right Original", pad_image2);
	imshow("Disparity", disp_image);
	imshow("Disparity2", simpoint_image);
	imshow("path", path);

	setMouseCallback("Disparity2", onMouseEvent, (void*)&simpoint_image);

	waitKey(0);

	for (int i = 0; i < width; i++)
		delete[] matrix[i];
	delete[] matrix; 
	for (int i = 0; i < width; i++)
		delete[] error_matrix[i];
	delete[] error_matrix;

	return 0;
}