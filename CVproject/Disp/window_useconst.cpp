#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <thread>

using namespace cv;
using namespace std;

#define CONST 60 // constraint 
#define OCCLUSIONCOST 400
#define PI 3.1415
#define LOW_TH 30
#define HIGH_TH 60

enum { DONE, DIAG, UP, LEFT };

// dynamic programming�� ����� ����ü
struct cost_case {
	double cost;
	uchar move;
	double dsi;
};

// edge detection�� ���� sobel filter ���
// arg = (source image, result sobel image, result sobel_direction array)
void Sobel_op(Mat& image, Mat& sobel_mag, int** sobel_ori) {
	int width = image.cols, height = image.rows;
	int x_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int y_filter[3][3] = { {1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } }; // 3 x 3 x���� sobel ����, y���� sobel ���� ����
	Mat pad_img(height + 4, width + 4, CV_8UC3), tmp_img(height, width, CV_8UC3); 
	int sum1, sum2, max = 0;
	float orientation;
	copyMakeBorder(image, pad_img, 1, 1, 1, 1, BORDER_REPLICATE); // padding for 3x3 filter (opencv �Լ� ���)
	cvtColor(pad_img, pad_img, COLOR_BGR2GRAY); // gray �̹����� ��ȯ (opencv �Լ� ���)

	// sobel operator
	// �̹����� sobel ���͸� convolution�ؼ� magnitude�� direction�� ���
	for (int y = 1; y < height + 1; y++) {
		for (int x = 1; x < width + 1; x++) {
			sum1 = 0;
			sum2 = 0;
			for (int n = -1; n <= 1; n++) {
				for (int m = -1; m <= 1; m++) { // 3x3�� sobel ���͸� �����ϸ� �� ���� ����
					sum1 += (pad_img.ptr<uchar>(y + n)[x + m] * x_filter[n + 1][m + 1]);
					sum2 += (pad_img.ptr<uchar>(y + n)[x + m] * y_filter[n + 1][m + 1]);
				}
			}
			// magnitude�� ���
			tmp_img.ptr<uchar>(y - 1)[x - 1] = sqrt(pow(sum1, 2) + pow(sum2, 2));
			max = max < sqrt(pow(sum1, 2) + pow(sum2, 2)) ? sqrt(pow(sum1, 2) + pow(sum2, 2)) : max; // 0~255 ������ scaling�ϱ� ���� max�� ����

			// ���⸦ 4�������� ��ȯ�Ͽ� edge normal ������ ����
			//  1  2  3
			//  4  *  4
			orientation = abs(atan2(sum2, sum1) * 180 / PI);
			if ((orientation >= (135 - 22.5)) && orientation < (135 + 22.5))
				sobel_ori[y - 1][x - 1] = 1;
			else if (orientation >= (90 - 22.5) && orientation < (90 + 22.5))
				sobel_ori[y - 1][x - 1] = 2;
			else if (orientation >= (45 - 22.5) && orientation < (45 + 22.5))
				sobel_ori[y - 1][x - 1] = 3;
			else if (orientation >= 0 && orientation < 22.5)
				sobel_ori[y - 1][x - 1] = 4;
			else if (orientation >= 135 + 22.5)
				sobel_ori[y - 1][x - 1] = 4;
		}
	}
	// magnitude�� 0~255 ������ ��ȯ
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			sobel_mag.ptr<uchar>(y)[x] = tmp_img.ptr<uchar>(y)[x] * 255 / max;
		}
	}
}

// arg = (source image, result canny image, sobel_direction array)
Mat canny_detc(Mat& image, Mat& img_canny, int ** sobel_ori) {
	int width = image.cols, height = image.rows;
	Mat img_gaussian, img_nonmax, sobel_mag(height, width, CV_8UC1);

	GaussianBlur(image, img_gaussian, Size(5, 5), 0); // Noise smoothing (opencv �Լ� ���)
	
	// sobel operator
	Sobel_op(img_gaussian, sobel_mag, sobel_ori);
	
	// Non-max suppression
	sobel_mag.copyTo(img_nonmax);
	uchar p, q;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1 ; x++) {
			uchar pvalue = sobel_mag.ptr<uchar>(y)[x]; // ���� �̹����� �ȼ����� ����.
			switch (sobel_ori[y][x]) { // �ش� �ȼ��� gradient ���� Ȯ�� �� gradient ������ ��,�� �ȼ����� ����
			case 1: // �»� ���� ����
				p = sobel_mag.ptr<uchar>(y - 1)[x - 1];
				q = sobel_mag.ptr<uchar>(y + 1)[x + 1];
				break;
			case 2: // �� �� ����
				p = sobel_mag.ptr<uchar>(y - 1)[x];
				q = sobel_mag.ptr<uchar>(y + 1)[x];
				break;
			case 3: // ��� ���� ����
				p = sobel_mag.ptr<uchar>(y - 1)[x + 1];
				q = sobel_mag.ptr<uchar>(y + 1)[x - 1];
				break;
			case 4: // �� �� ����
				p = sobel_mag.ptr<uchar>(y)[x - 1];
				q = sobel_mag.ptr<uchar>(y)[x + 1];
				break;
			}
			if(!(pvalue >= p && pvalue >= q)) 
				img_nonmax.ptr<uchar>(y)[x] = 0; // case(gradient ����) ���� ��, �� �ȼ����� ���ؼ� nonmax�� ��� �ȼ����� 0���� ����
		}
	}
	
	// double threshold
	Mat img_high(height, width, CV_8UC1), img_low(height, width, CV_8UC1);
	// �ΰ��� threshold�� ����� �ΰ��� edge map�� ����
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img_nonmax.ptr<uchar>(y)[x] > HIGH_TH)
				img_high.ptr<uchar>(y)[x] = 255;
			else
				img_high.ptr<uchar>(y)[x] = 0;

			if (img_nonmax.ptr<uchar>(y)[x] > LOW_TH)
				img_low.ptr<uchar>(y)[x] = 255;
			else
				img_low.ptr<uchar>(y)[x] = 0;
		}
	}
	img_high.copyTo(img_canny); // edge�� ������ �̹��� ����

	// Edge connecting
	int yn, xm, count;
	bool do_break = false;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			yn = y;
			xm = x;
			do_break = false;
			if (img_high.ptr<uchar>(y)[x] == 255 || img_canny.ptr<uchar>(y)[x] == 255) { // high�� �ִ� edge�̰ų� low�� ����Ǿ��־� �߰��� edge�� ���
				count = 0; // Ȯ���� �ȼ� �� ����(�ݺ��� ���Ḧ ����)
				while (count != 25) { // 5x5 ��� �ȼ��� �� ä����� ���������� �ݺ�
					count = 0;
					do_break = true; // �ȼ��� ã���� ���߱� ���� ����
					for (int n = -1; n <= 1; n++) { // ó������ 3x3 �� �̿��ؼ� ����� �ȼ��� ã������ �ʹ� ª�Ƽ� 5x5 �߰�
						for (int m = -1; m <= 1; m++) { // �ֺ��� low�� ���� edge�� �ְ� ���� ĥ������ �ʾ����� ĥ�ϰ� �� �ȼ��� �̵�
							if (xm + m > width - 1 || xm + m < 0 || yn + n > height - 1 || yn + n < 0) 
								break; // �̹��� ���� �ʰ��� ����
							if ((img_low.ptr<uchar>(yn + n)[xm + m] == 255) && (img_canny.ptr<uchar>(yn + n)[xm + m] == 0)) {
								img_canny.ptr<uchar>(yn + n)[xm + m] = 255; // high�� ������ִ� low edge�ε� �߰����� ���� ��� edge�� �߰�
								yn = y + n; 
								xm = x + m; // ���� �߰��� edge �ȼ��� �̵�(���⼭ �� ã�� ����)
								do_break = false; 
								break; // �̵��� �ȼ����� �ٽ� ó������ ã������ �ݺ��� ����
							}
						}
						if (!do_break) 
							break; // �̵��� �ȼ����� �ٽ� ó������ ã������ �ݺ��� ����
					}
					if (do_break) {
						for (int n = -2; n <= 2; n++) { // �ֺ� 3x3�� �������� 5x5 �� Ȯ���ؼ� ����� �ȼ��� ã��. ������ window ũ�� ���� 3x3 �κа� ����
							for (int m = -2; m <= 2; m++) { // �ֺ��� low�� ���� edge�� �ְ� ���� ĥ������ �ʾ����� ĥ�ϰ� �� �ȼ��� �̵�
								if (xm + m > width - 2 || xm + m < 0 || yn + n > height - 2 || yn + n < 0) {
									count++;
								}
								else if ((img_low.ptr<uchar>(yn + n)[xm + m] == 255) && (img_canny.ptr<uchar>(yn + n)[xm + m] == 0)) {
									img_canny.ptr<uchar>(yn + n)[xm + m] = 255;
									yn = y + n;
									xm = x + m;
									do_break = false;
									break;
								}
								else
									count++;
							}
							if (!do_break)
								break;
						} 
					}
				}
			}
		}
	}
	
	//imwrite("nonmax.jpg", img_nonmax);
	imwrite("img_high.jpg", img_high);
	imwrite("img_low.jpg", img_low);
	imwrite("img_canny.jpg", img_canny);
	//imshow("img_high.jpg", img_high);
	//imshow("img_low.jpg", img_low);
	//imshow("img_canny.jpg", img_canny);
	//waitKey(0);
	
	return img_canny;
}

// arg = (left image, right image, result disparity image, matching window size, thread numer 1 to 3)
void StereoMatching_DP_1_3(Mat& image1, Mat& image2, Mat& disparity_image, int win_size, int num) {

	int win_parm = win_size / 2;

	int width = image1.cols;
	int height = image1.rows;

	Mat pad_image1(height + win_parm * 2, width + win_parm * 2, CV_8UC3); // pading image
	Mat pad_image2(height + win_parm * 2, width + win_parm * 2, CV_8UC3); // pading image

	// padding for 5x5 filter (opencv �Լ� ���)
	copyMakeBorder(image1, pad_image1, win_parm, win_parm, win_parm, win_parm, BORDER_REPLICATE);
	copyMakeBorder(image2, pad_image2, win_parm, win_parm, win_parm, win_parm, BORDER_REPLICATE);

	Mat disp_image(width, width, CV_8UC1); // DSI image
	//Mat path(width, width, CV_8UC1); // ��� image. ���������� ������� ����

	// DP�� ����� cost�� �̵� case�� ������ matrix ����
	cost_case** matrix = new cost_case * [width + 1];
	for (int i = 0; i < width + 1; i++)
		matrix[i] = new cost_case[width + 1];

	double* errorbox = new double[width]; // max���� ã������ error���� ������ array

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

	// DSI ���ϱ�
	for (int y = height * (num - 1) / 3 + win_parm; y < height * num / 3 + win_parm; y++) { // thread�� y ���� 1/3�� ����
		disp_image.setTo(0);
		//path.setTo(255);
		// y�� DSI�� ���ϰ� DP
		for (int x = win_parm; x < width + win_parm; x++) {
			uchar* pointer_output = disp_image.ptr<uchar>(x - win_parm); // DSI �̹��� �ּҸ� �����Ϳ� ����
			double max = 0, error;
			for (int k = win_parm; k < width + win_parm; k++) {
				error = 0;
				int l = k - win_parm;
				if (x < CONST) { // ī�޶� �̵� ��ġ�� ���� ��ġ ��ȭ�� �Ѱ谡 �����Ƿ� �� ���� �̻��� ������� ����
					if (l < x) {
						for (int winh = -win_parm; winh <= win_parm; winh++) { // 5 * 5 window�� �̿��� error�� ���
							for (int winv = -win_parm; winv <= win_parm; winv++) {
								uchar b1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 0];
								uchar g1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 1];
								uchar r1 = pad_image1.ptr<uchar>(y + winv)[(x + winh) * 3 + 2];
								uchar b2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 0];
								uchar g2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 1];
								uchar r2 = pad_image2.ptr<uchar>(y + winv)[(k + winh) * 3 + 2];
								error += (pow(b1 - b2, 2) + pow(g1 - g2, 2) + pow(r1 - r2, 2));
							}
						}
						errorbox[l] = error; // ���߿� error���� ���� ������ ���� ��Ƶ�
						matrix[x - win_parm + 1][l + 1].dsi = error / 50; // error���� �ʹ� Ŀ�� ���� ����
						max = (max < error) ? error : max;
					}
					else { // ���� ������ �Ѿ �κ��� error���� ��Ⱚ�� ũ�� �༭ �ǹ� ������ ��
						pointer_output[l] = 255;
						matrix[x - win_parm + 1][l + 1].dsi = 100000; 
					}
				}
				else { // ���� ������ �����ϰ�� ���� ����
					if (l < x - CONST) {
						pointer_output[l] = 255;
						matrix[x - win_parm + 1][l + 1].dsi = 100000;
					}
					else if ((x - CONST < l) && (l < x - 1)) {
						for (int winh = -win_parm; winh <= win_parm; winh++) { // 5 * 5 window dis
							for (int winv = -win_parm; winv <= win_parm; winv++) {
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
						matrix[x - win_parm + 1][l + 1].dsi = error / 50;
						max = (max < error) ? error : max;
					}
					else {
						pointer_output[l] = 255;
						matrix[x - win_parm + 1][l + 1].dsi = 100000;
					}
				}
			}

			for (int k = 0; k < width; k++) { // error���� ������ 0 ~ 255 ������ �ٲٱ� ���ؼ� ����(DSI �̹����� Ȯ���ϱ� ���ؼ� ��������)
				if (pointer_output[k] != 255) // ���� ������ ��� �κ��� 255�� ���������Ƿ� ����
					pointer_output[k] = errorbox[k] / max * 255;
			}

			if (x > win_parm) // �迭�� DP�� ���� �� �ȼ������� cost�� move ����
				for (int m = 1; m < width + 1; m++) {
					int min1 = matrix[x - win_parm - 1][m - 1].cost + matrix[x - win_parm][m].dsi;
					int min2 = matrix[x - win_parm - 1][m].cost + OCCLUSIONCOST;
					int min3 = matrix[x - win_parm][m - 1].cost + OCCLUSIONCOST;

					int lastmin = min(min1, min2);
					lastmin = min(min3, lastmin); // 3���� ����� �ּҰ��� ����

					matrix[x - win_parm][m].cost = lastmin;
					if (min1 == lastmin) matrix[x - win_parm][m].move = DIAG;
					if (min3 == lastmin) matrix[x - win_parm][m].move = LEFT;
					if (min2 == lastmin) matrix[x - win_parm][m].move = UP; // �ּҰ��� ������ ��ΰ� ����� ����
				}
		}
		for (int m = 1; m < width + 1; m++) { // ������ ���� ������ ������� �ʰ� ������ ������ ������ �� ���
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
		uchar* pointer_simimage = disparity_image.ptr<uchar>(y - win_parm); // disparity image y point
		// ������ �Ʒ� ������ �ǵ��ƿ��鼭 ��� ã��
		int p = width - 1, q = width - 1;
		int point = p;
		while (p != 0 && q != 0) { // ���� �� ���� �����ϸ� ����
			switch (matrix[p][q].move) { // ������ �����ص� ��θ� ���� �ö�. �� case�� �´� ���� disparity image�� ĥ�� -> ������ disparity �̹����� ������ ĥ����
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
			//path.at<uchar>(p, q) = 0; // path�� ���� Ȯ���ϱ����� ���������� ĥ��. ������ path �̹����� ���� �������� ����
		}
		// ������ ���
		if ((float)y / (height + win_parm) * 100 < 33.4) {
			system("cls");
			cout << " processing " << (float)y / (height + win_parm) * 100 * 3 << "%" << endl;
		}
		//imshow("path", path);
		//waitKey(0);
	}
	for (int i = 0; i < width; i++)
		delete[] matrix[i];
	delete[] matrix;
}

// arg = (stereo image, canny edge image, sobel_direction array)
Mat Fill_with_edge(const Mat& stereo, const Mat& edge, int** sobel_ori) {
	int height = stereo.rows, width = stereo.cols, dir, pixel;
	bool first;
	Mat stereo_pad(height + 4, width + 4, CV_8UC1); // for pading
	copyMakeBorder(stereo, stereo_pad, 2, 2, 2, 2, BORDER_ISOLATED);
	
	for (int y = 2; y < height + 2; y++) {
		for (int x = 2; x < width + 2; x++) {
			pixel = 0;
			first = true;
			if (edge.ptr<uchar>(y - 2)[x - 2] == 255) { // �ȼ��� edge���
				switch (sobel_ori[y - 2][x - 2]) // �� �ȼ��� direction�� �°� ���� ä���
				{
				case 1:
					// �»�ܺκ� ä���
					for (int i = -2; i <= 2; i++) {
						for (int j = -2; j <= -i; j++) {
							// �� �ȼ��� window���� ĥ���� ���� ã�°Ÿ�
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) { // ĥ���� ���� ã����
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j]; // �� ���� �����ϰ�
									j = -2;
									i = -3;
									first = false; // �ݺ����� �� ó������ �ٽ� ������.
									break;
								}
							}
							// �� �ȼ��� window���� ���� ã�� �ٽ� ó������ �°Ÿ�
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0) // hole �κ��� ���� �Ʊ� ã�� ������ ä���.
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j]; // ���ο� ���� ã���� �׶����ʹ� �� ������ ä���.
								}
							} // �Ʒ� case ��� edge�� ���⿡ ���� �� ����� �ݺ�
						}
					}
					// ���ϴܺκ� ä���. ����� ���� ����
					for (int i = -1; i <= 2; i++) {
						for (int j = 1 - i; j < 2; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = 2;
									i = -2;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					break;
				case 2: // �Ʒ����ʹ� ��� case 1�� ���� ��������� ���⿡ ���� ä��� ���� �ٸ�.
					// ��� ä���
					for (int i = -2; i <= 0; i++) {
						for (int j = -2; j <= 2; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = -2;
									i = -3;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					// �ϴ� ä���
					for (int i = 1; i <= 2; i++) {
						for (int j = -2; j <= 2; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = -2;
									i = -3;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					break;
				case 3:
					// ���ܺκ� ä���
					for (int i = -2; i <= 2; i++) {
						for (int j = i; j <= 2; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = -2;
									i = -3;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					// ���ϴܺκ� ä���
					for (int i = -2; i <= 2; i++) {
						for (int j = -2; j < i - 1; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = 2;
									i = -2;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					break;
				case 4:
					// ���� ä���
					for (int i = -2; i <= 2; i++) {
						for (int j = -2; j <= 0; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = -2;
									i = -3;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					// ���� ä���
					for (int i = -2; i <= 2; i++) {
						for (int j = 1; j <= 2; j++) {
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
									j = 2;
									i = -2;
									first = false;
									break;
								}
							}
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0)
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j];
								}
							}
						}
					}
					break;
				}
			}
		}
	}
	
	// delete padding part
	Rect rect(Point(2, 2), Point(width + 2, height + 2));
	Mat stereo_fill = stereo_pad(rect);

	// occlusion filling (��ȿ�� ���� pixel������ ä���)
	for (int y = 0; y < height; y++) {
		uchar value = 0;
		uchar* pointer = stereo_fill.ptr<uchar>(y);
		for (int x = 0; x < width; x++) {
			if (pointer[x]) {
				value = pointer[x];
			}
			else
				pointer[x] = value;
		}
	}
	return stereo_fill;
}


int main()
{
	int win_size = 5; // set stereo matching window size
	// left image
	Mat image1 = imread("iml.png", IMREAD_COLOR);
	if (image1.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	// right image
	Mat image2 = imread("imr.png", IMREAD_COLOR);
	if (image2.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	int width = image1.cols;
	int height = image1.rows;
	Mat disparity_image(height, width, CV_8UC1); // disparity image
	Mat canny_edge, disparity_image_fill(height, width, CV_8UC1), disparity_image_fill_useedge;
	disparity_image.setTo(0);

	int** sobel_ori = new int* [height]; // sobel ���͸� ����ؼ� edge�� ������ �����ϱ� ���� �迭
	for (int i = 0; i < height; i++)
		sobel_ori[i] = new int[width];

	canny_detc(image1, canny_edge, sobel_ori); // edge detection
	
	// depth map ���� (�ӵ� ���̱� ���ؼ� multi thread ���)
	thread t1(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 1);
	thread t2(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 2);
	thread t3(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 3);
	t1.join();
	t2.join();
	t3.join();

	// occlusion filling (��ȿ�� ���� pixel������ ä���)
	for (int y = 0; y < height; y++) {
		uchar* pointer_nfill = disparity_image.ptr<uchar>(y);
		uchar* pointer_fill = disparity_image_fill.ptr<uchar>(y);
		uchar pixel = 0;
		for (int x = 0; x < width; x++) {
			if (pointer_nfill[x]) { // ���� disparity map�� ���� �ȼ��� hole�� �ƴϸ�
				pixel = pointer_nfill[x]; // �� ���� �����صΰ�
				pointer_fill[x] = pixel; // ���� ��ġ�� �� ������ ä���
			}
			else
				pointer_fill[x] = pixel; // ���� disparity map�� ���� �ȼ��� hole�̸� ������ �����ص� ������ ä���
		}
	}

	// edge ������ Ȱ���ؼ� filling �ϱ�
	disparity_image_fill_useedge = Fill_with_edge(disparity_image, canny_edge, sobel_ori);

	//imshow("left Original", pad_image1);
	//imshow("right Original", pad_image2);
	//imshow("DSI", disp_image);
	imshow("result of DP", disparity_image);
	imshow("result of DP with filling", disparity_image_fill);
	imshow("result of DP with filling use edge", disparity_image_fill_useedge);
	//imshow("path", path);
	imwrite("result of DP.jpg", disparity_image);
	imwrite("result of DP with filling.jpg", disparity_image_fill);
	imwrite("result of DP with filling use edge.jpg", disparity_image_fill_useedge);

	waitKey(0);

	for (int i = 0; i < height; i++)
		delete[] sobel_ori[i];
	delete[] sobel_ori;

	return 0;
}