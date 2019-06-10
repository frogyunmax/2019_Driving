## 자율주행 차량의 차선변경 판단 알고리즘
#### Algorithm for determining the car lane change of Auto-Driving Car

[2019 1학기 정보과 과제연구] 한성과학고등학교 2107 김서현, 2219 이현수

***
### [1] 연구의 목적

본 연구의 최종 목적은, 자율주행 자동차에 적용가능한 실용가능성이 있는 자율주행 알고리즘을 제작하는 것이다. OpenCV를 이용한 차선인식 알고리즘과 YOLO Argorithm을 이용해 시간에 따른 좌표값 변화율로 상대속도를 측정할 수 있는 알고리즘을 제작하였다. 이에 대해 실용가능성이 있는 더 나은 자율주행 알고리즘 제작을 위해, 기존 자율주행 알고리즘의 문제점을 발견하고 이를 개선하여 최종적으로 자율주행 차량의 차선변경 판단 알고리즘을 수립하였다.

![noname01](https://user-images.githubusercontent.com/40256530/59174919-52a9fe00-8b8e-11e9-8880-2995e8d28ad4.png)

### [2] 이론적 배경

- OpenCv : 차선 인식을 위해
   - Hough Transform, Probabilistic Hough Transform, Gaussian Blur, Canny Edge Detection, etc.

- YOLO : 차량 인식 및 추적을 위해

### [3] 연구 방법 및 결과

1) 차선 인식 알고리즘 개선

   - 차선 종류 인식기능 
   
   - 도로 주변 물체 (가로동 등의) 인식 문제 개선
   
   - 그림자가 있는 경우 차선인식 효율 개선 : 차선 부분 Amplify 시키기
   
   ![noname02](https://user-images.githubusercontent.com/40256530/59175100-36f32780-8b8f-11e9-961d-21ab63903982.png)

2) YOLO 알고리즘을 통한 차량 간 상대 속도 측정

   - 변경하려고 하는 차선 범위 내의 차량만 인식
   
   - 인식한 차량 중 차선변경 시 고려할 차량만을 추적
   
   - 상대속도 (좌표 변화량) 측정
   
   ![noname03](https://user-images.githubusercontent.com/40256530/59175182-86d1ee80-8b8f-11e9-9f6b-d3be1aba3220.JPG)
   
### [4] 참고자료

- https://github.com/tawnkramer/CarND-Vehicle-Detection

- https://github.com/tawnkramer/CarND-Advanced-Lane-Lines

- https://www.youtube.com/watch?v=eLTLtUVuuy4&list=PLAY_IJkbyNZN6X31kkKqujo7EIyBqMlvZ&index=4&t=0s

- Datasets from Kaggle and Youtube

***
