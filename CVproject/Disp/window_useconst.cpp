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

// dynamic programming에 사용할 구조체
struct cost_case {
	double cost;
	uchar move;
	double dsi;
};

// edge detection을 위해 sobel filter 사용
// arg = (source image, result sobel image, result sobel_direction array)
void Sobel_op(Mat& image, Mat& sobel_mag, int** sobel_ori) {
	int width = image.cols, height = image.rows;
	int x_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int y_filter[3][3] = { {1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } }; // 3 x 3 x방향 sobel 필터, y방향 sobel 필터 생성
	Mat pad_img(height + 4, width + 4, CV_8UC3), tmp_img(height, width, CV_8UC3); 
	int sum1, sum2, max = 0;
	float orientation;
	copyMakeBorder(image, pad_img, 1, 1, 1, 1, BORDER_REPLICATE); // padding for 3x3 filter (opencv 함수 사용)
	cvtColor(pad_img, pad_img, COLOR_BGR2GRAY); // gray 이미지로 변환 (opencv 함수 사용)

	// sobel operator
	// 이미지와 sobel 필터를 convolution해서 magnitude와 direction를 계산
	for (int y = 1; y < height + 1; y++) {
		for (int x = 1; x < width + 1; x++) {
			sum1 = 0;
			sum2 = 0;
			for (int n = -1; n <= 1; n++) {
				for (int m = -1; m <= 1; m++) { // 3x3의 sobel 필터를 적용하며 그 값을 저장
					sum1 += (pad_img.ptr<uchar>(y + n)[x + m] * x_filter[n + 1][m + 1]);
					sum2 += (pad_img.ptr<uchar>(y + n)[x + m] * y_filter[n + 1][m + 1]);
				}
			}
			// magnitude를 계산
			tmp_img.ptr<uchar>(y - 1)[x - 1] = sqrt(pow(sum1, 2) + pow(sum2, 2));
			max = max < sqrt(pow(sum1, 2) + pow(sum2, 2)) ? sqrt(pow(sum1, 2) + pow(sum2, 2)) : max; // 0~255 범위로 scaling하기 위해 max값 저장

			// 기울기를 4방향으로 변환하여 edge normal 방향을 저장
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
	// magnitude를 0~255 범위로 변환
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

	GaussianBlur(image, img_gaussian, Size(5, 5), 0); // Noise smoothing (opencv 함수 사용)
	
	// sobel operator
	Sobel_op(img_gaussian, sobel_mag, sobel_ori);
	
	// Non-max suppression
	sobel_mag.copyTo(img_nonmax);
	uchar p, q;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1 ; x++) {
			uchar pvalue = sobel_mag.ptr<uchar>(y)[x]; // 기존 이미지의 픽셀값을 저장.
			switch (sobel_ori[y][x]) { // 해당 픽셀의 gradient 방향 확인 후 gradient 방향의 앞,뒤 픽셀값도 저장
			case 1: // 좌상 우하 방향
				p = sobel_mag.ptr<uchar>(y - 1)[x - 1];
				q = sobel_mag.ptr<uchar>(y + 1)[x + 1];
				break;
			case 2: // 상 하 방향
				p = sobel_mag.ptr<uchar>(y - 1)[x];
				q = sobel_mag.ptr<uchar>(y + 1)[x];
				break;
			case 3: // 우상 좌하 방향
				p = sobel_mag.ptr<uchar>(y - 1)[x + 1];
				q = sobel_mag.ptr<uchar>(y + 1)[x - 1];
				break;
			case 4: // 좌 우 방향
				p = sobel_mag.ptr<uchar>(y)[x - 1];
				q = sobel_mag.ptr<uchar>(y)[x + 1];
				break;
			}
			if(!(pvalue >= p && pvalue >= q)) 
				img_nonmax.ptr<uchar>(y)[x] = 0; // case(gradient 방향) 마다 앞, 뒤 픽셀값과 비교해서 nonmax일 경우 픽셀값을 0으로 변경
		}
	}
	
	// double threshold
	Mat img_high(height, width, CV_8UC1), img_low(height, width, CV_8UC1);
	// 두가지 threshold를 사용해 두개의 edge map을 생성
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
	img_high.copyTo(img_canny); // edge를 연결할 이미지 복사

	// Edge connecting
	int yn, xm, count;
	bool do_break = false;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			yn = y;
			xm = x;
			do_break = false;
			if (img_high.ptr<uchar>(y)[x] == 255 || img_canny.ptr<uchar>(y)[x] == 255) { // high에 있는 edge이거나 low와 연결되어있어 추가된 edge일 경우
				count = 0; // 확인한 픽셀 수 저장(반복분 종료를 위해)
				while (count != 25) { // 5x5 모든 픽셀에 더 채울곳이 없을때까지 반복
					count = 0;
					do_break = true; // 픽셀을 찾으면 멈추기 위한 변수
					for (int n = -1; n <= 1; n++) { // 처음에는 3x3 을 이용해서 연결된 픽셀을 찾았지만 너무 짧아서 5x5 추가
						for (int m = -1; m <= 1; m++) { // 주변에 low를 넘은 edge가 있고 아직 칠해지지 않았으면 칠하고 그 픽셀로 이동
							if (xm + m > width - 1 || xm + m < 0 || yn + n > height - 1 || yn + n < 0) 
								break; // 이미지 범위 초과시 멈춤
							if ((img_low.ptr<uchar>(yn + n)[xm + m] == 255) && (img_canny.ptr<uchar>(yn + n)[xm + m] == 0)) {
								img_canny.ptr<uchar>(yn + n)[xm + m] = 255; // high에 연결돼있는 low edge인데 추가되지 않은 경우 edge로 추가
								yn = y + n; 
								xm = x + m; // 이후 추가된 edge 픽셀로 이동(여기서 또 찾기 위해)
								do_break = false; 
								break; // 이동한 픽셀에서 다시 처음부터 찾기위해 반복분 종료
							}
						}
						if (!do_break) 
							break; // 이동한 픽셀에서 다시 처음부터 찾기위해 반복분 종료
					}
					if (do_break) {
						for (int n = -2; n <= 2; n++) { // 주변 3x3에 없었을때 5x5 을 확인해서 연결된 픽셀을 찾기. 내용은 window 크기 제외 3x3 부분과 동일
							for (int m = -2; m <= 2; m++) { // 주변에 low를 넘은 edge가 있고 아직 칠해지지 않았으면 칠하고 그 픽셀로 이동
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

	// padding for 5x5 filter (opencv 함수 사용)
	copyMakeBorder(image1, pad_image1, win_parm, win_parm, win_parm, win_parm, BORDER_REPLICATE);
	copyMakeBorder(image2, pad_image2, win_parm, win_parm, win_parm, win_parm, BORDER_REPLICATE);

	Mat disp_image(width, width, CV_8UC1); // DSI image
	//Mat path(width, width, CV_8UC1); // 경로 image. 최종본에는 사용하지 않음

	// DP에 사용할 cost와 이동 case를 저장할 matrix 선언
	cost_case** matrix = new cost_case * [width + 1];
	for (int i = 0; i < width + 1; i++)
		matrix[i] = new cost_case[width + 1];

	double* errorbox = new double[width]; // max값을 찾기위해 error값을 저장할 array

	// matrix 초기 설정
	for (int i = 0; i < width + 1; i++) {
		for (int j = 0; j < width + 1; j++) {
			if (i == 0) { // 첫행 cost, move 설정
				matrix[i][j].cost = OCCLUSIONCOST * j;
				if (j == 0)
					matrix[i][j].move = DONE;
				else
					matrix[i][j].move = LEFT;
			}
			else {
				if (j == 0) { // 첫열 cost, move 설정
					matrix[i][j].cost = OCCLUSIONCOST * i;
					matrix[i][j].move = UP; // UP -> DIAG
				}
			}
		}
	}

	// DSI 구하기
	for (int y = height * (num - 1) / 3 + win_parm; y < height * num / 3 + win_parm; y++) { // thread별 y 범위 1/3씩 수행
		disp_image.setTo(0);
		//path.setTo(255);
		// y열 DSI를 구하고 DP
		for (int x = win_parm; x < width + win_parm; x++) {
			uchar* pointer_output = disp_image.ptr<uchar>(x - win_parm); // DSI 이미지 주소를 포인터에 저장
			double max = 0, error;
			for (int k = win_parm; k < width + win_parm; k++) {
				error = 0;
				int l = k - win_parm;
				if (x < CONST) { // 카메라 이동 위치에 따라 위치 변화의 한계가 있으므로 이 차이 이상을 고려하지 않음
					if (l < x) {
						for (int winh = -win_parm; winh <= win_parm; winh++) { // 5 * 5 window를 이용해 error값 계산
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
						errorbox[l] = error; // 나중에 error값의 범위 조절을 위해 모아둠
						matrix[x - win_parm + 1][l + 1].dsi = error / 50; // error값이 너무 커서 조금 줄임
						max = (max < error) ? error : max;
					}
					else { // 제한 범위를 넘어간 부분은 error값과 밝기값을 크게 줘서 의미 없도록 함
						pointer_output[l] = 255;
						matrix[x - win_parm + 1][l + 1].dsi = 100000; 
					}
				}
				else { // 제한 범위를 제외하고는 위와 동일
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

			for (int k = 0; k < width; k++) { // error값의 범위를 0 ~ 255 값으로 바꾸기 위해서 수행(DSI 이미지를 확인하기 위해서 수행했음)
				if (pointer_output[k] != 255) // 제한 범위를 벗어난 부분은 255로 설정했으므로 무시
					pointer_output[k] = errorbox[k] / max * 255;
			}

			if (x > win_parm) // 배열에 DP를 위해 각 픽셀에서의 cost와 move 저장
				for (int m = 1; m < width + 1; m++) {
					int min1 = matrix[x - win_parm - 1][m - 1].cost + matrix[x - win_parm][m].dsi;
					int min2 = matrix[x - win_parm - 1][m].cost + OCCLUSIONCOST;
					int min3 = matrix[x - win_parm][m - 1].cost + OCCLUSIONCOST;

					int lastmin = min(min1, min2);
					lastmin = min(min3, lastmin); // 3개의 경로중 최소값을 결정

					matrix[x - win_parm][m].cost = lastmin;
					if (min1 == lastmin) matrix[x - win_parm][m].move = DIAG;
					if (min3 == lastmin) matrix[x - win_parm][m].move = LEFT;
					if (min2 == lastmin) matrix[x - win_parm][m].move = UP; // 최소값을 가지는 경로가 어딘지 저장
				}
		}
		for (int m = 1; m < width + 1; m++) { // 마지막 행은 위에서 계산하지 않고 끝나기 때문에 마지막 행 계산
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
		// 오른쪽 아래 끝부터 되돌아오면서 경로 찾기
		int p = width - 1, q = width - 1;
		int point = p;
		while (p != 0 && q != 0) { // 왼쪽 위 끝에 도착하면 종료
			switch (matrix[p][q].move) { // 위에서 저장해둔 경로를 따라 올라감. 각 case에 맞는 값을 disparity image에 칠함 -> 끝나면 disparity 이미지의 한줄이 칠해짐
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
			//path.at<uchar>(p, q) = 0; // path를 직접 확인하기위해 검은색으로 칠함. 지금은 path 이미지를 따로 저장하지 않음
		}
		// 진행율 출력
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
			if (edge.ptr<uchar>(y - 2)[x - 2] == 255) { // 픽셀이 edge라면
				switch (sobel_ori[y - 2][x - 2]) // 그 픽셀의 direction에 맞게 주위 채우기
				{
				case 1:
					// 좌상단부분 채우기
					for (int i = -2; i <= 2; i++) {
						for (int j = -2; j <= -i; j++) {
							// 그 픽셀의 window에서 칠해진 값을 찾는거면
							if (first) {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] != 0) { // 칠해진 값을 찾으면
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j]; // 그 색을 저장하고
									j = -2;
									i = -3;
									first = false; // 반복문을 맨 처음부터 다시 돌린다.
									break;
								}
							}
							// 그 픽셀의 window에서 값을 찾고 다시 처음부터 온거면
							else {
								if (stereo_pad.ptr<uchar>(y + i)[x + j] == 0) // hole 부분의 색을 아까 찾은 색으로 채운다.
									stereo_pad.ptr<uchar>(y + i)[x + j] = pixel;
								else {
									pixel = stereo_pad.ptr<uchar>(y + i)[x + j]; // 새로운 값을 찾으면 그때부터는 이 색으로 채운다.
								}
							} // 아래 case 모두 edge의 방향에 따라 위 방법을 반복
						}
					}
					// 우하단부분 채우기. 방법은 위와 동일
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
				case 2: // 아래부터는 모두 case 1과 같은 방법이지만 방향에 따라 채우는 곳만 다름.
					// 상단 채우기
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
					// 하단 채우기
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
					// 우상단부분 채우기
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
					// 좌하단부분 채우기
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
					// 좌측 채우기
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
					// 우측 채우기
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

	// occlusion filling (유효한 이전 pixel값으로 채우기)
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

	int** sobel_ori = new int* [height]; // sobel 필터를 사용해서 edge의 방향을 저장하기 위한 배열
	for (int i = 0; i < height; i++)
		sobel_ori[i] = new int[width];

	canny_detc(image1, canny_edge, sobel_ori); // edge detection
	
	// depth map 생성 (속도 높이기 위해서 multi thread 사용)
	thread t1(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 1);
	thread t2(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 2);
	thread t3(StereoMatching_DP_1_3, ref(image1), ref(image2), ref(disparity_image), win_size, 3);
	t1.join();
	t2.join();
	t3.join();

	// occlusion filling (유효한 이전 pixel값으로 채우기)
	for (int y = 0; y < height; y++) {
		uchar* pointer_nfill = disparity_image.ptr<uchar>(y);
		uchar* pointer_fill = disparity_image_fill.ptr<uchar>(y);
		uchar pixel = 0;
		for (int x = 0; x < width; x++) {
			if (pointer_nfill[x]) { // 구한 disparity map중 지금 픽셀이 hole이 아니면
				pixel = pointer_nfill[x]; // 그 값을 저장해두고
				pointer_fill[x] = pixel; // 지금 위치도 그 값으로 채우고
			}
			else
				pointer_fill[x] = pixel; // 구한 disparity map중 지금 픽셀이 hole이면 이전에 저장해둔 값으로 채운다
		}
	}

	// edge 정보를 활용해서 filling 하기
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