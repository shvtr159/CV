# Computer Vsion project
## Dynamic programming을 이용한 Disparity image 만들기
이번 프로젝트에서는 객체의 깊이를 알아내고 Depth map을 완성하기 위해단안 단서 중 중첩과 운동 시차를 이용하여 이미지의 depth 정보를 얻는다.
## 프로젝트 내용
### 1. 중첩
우리는 어떠한 물체가 뒤의 물체를 가린 것인지, 아니면 같은 물체인 것인지 판단하기 위해 물체의 색의 변화, 패턴의 변화 등을 이용하여 그 형태를 파악한다. 형태를 파악하고 나면 각각의 물체가 차지하고 있는 영역을 알 수 있고 이를 이용해 물체의 앞뒤 관계를 파악할 수 있다. occlusion으로 인해 생긴 hole을 채우는 방법에 이 원리를 이용하기 위해 이미지에서 물체의 edge를 찾아낸다.
edge를 찾아내기 위해 canny edge detector를 사용한다. 이는 다음 3단계를 거쳐 edge를 찾아낸다.
1. Noise smoothing
2. Nonmax-suppression
3. Hysteresis threshoding

### 2. 운동시차
1. stereo image를 이용해서 DSI(Disparity Space Image)를 생성한다. 
2. Dynamic programming을 이용해 DSI에서의 최적의 경로를 찾는다.
3. 이를 이용해 disparity image를 구한다.

### 3. Hole filling
1번과 2번의 결과를 합쳐 검게 나타나는 부분들을 적절하게 채운다.
Edge가 완벽히 물체의 외부를 표현하고 있다면 이 edge를 기준으로 물체가 달라지거나 배경이 나타나므로 disparity 값이 달라질 것이다. 이 가정을 이용해 hole filling에 edge를 사용한다. Disparity 이미지와 edge 이미지의 픽셀을 같이 움직이며 edge나 나타나면 그 픽셀의 gradient direction을 확인하고 이에 맞게 Disparity 이미지를 참고하여 색을 채운다.

## 프로젝트 결과
눈으로 보기에는 프로젝트 결과 이미지가 표면이 더 울퉁불퉁하여 전체적으로 성능이 나빠 보이지만 왼쪽의 occlusion된 부분을 채우고 물체의 edge에 관련된 부분을 좀 더 살리고 있다. 이 결과를 정량적으로 확인하기 위해 ground truth 이미지와의 MSSIM(Mean Structural Similarity)를 측정하였다. 설정값은 기본 아래와 같이 설정되어있다.

|      |Low threshold       |High threshold | Matcing window size|
|-------|-------------------|---------------|--------------|
|**value** |  30            |60             |     5x5      |

변화가 큰 부분의 영역을 측정한 결과
|    | cones | mask | whole |
|----|-------|------|-------|
|simple filling|0.789874|0.859588|0.739214|
|project result|0.792855|0.858217|0.75585|

콘 부분은 edge를 이용해 프로젝트의 결과가 조금 더 콘들의 모양에 가까워졌지만 가면 부분은 occlusion 된 부분이 많이 없었고, 그중 원래 값과 다르게 채워진 pixel로 인해 edge 너머의 부분까지 채워지며 원본과 달라져 단순 filling의 효과가 좀 더 좋게 나타났다. 그러나 전체 이미지로 보았을 때는 프로젝트의 결과가 조금 더 좋게 나온 것을 알 수 있다.