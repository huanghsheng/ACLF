close all;clear;clc;

set(0,'defaultfigurecolor','w') 
DistortFlag = 0; %input('Is there distortion of infrared image? :\n');
[I1gray, I2gray, I1rgb, I2rgb, f1, f2, path] = cp_readImage;
%% section II: Resize images based on the minimum imaclosege height
height = size(I1gray,1);
[I1, I2, scale] = cp_resizeImage(I1gray,I2gray,height);

P1 = load('matrix1.mat');
P1=P1.matrix;
P2 = load('matrix2.mat');
P2=P2.matrix;
P3 = cp_subpixelFine(P1,P2); % Fine matching
% P3=P2;
% toc
%% section V: Show visual registration result 就是两个和在�?��的效�?
[~,affmat] = cp_getAffine(I1gray,I2gray,P1,P3);
Imosaic = cp_graymosaic(I1gray, I2gray, affmat);
figure, subplot(121),imshow(Imosaic);subplot(122),imshow(cp_rgbmosaic(I1rgb,I2rgb,affmat)); %如果是彩色图像的话运行后面这�?
cp_showResult(I1rgb,I2rgb,I1gray,I2gray,affmat,3); % checkborder image
cp_showMatch(I1rgb,I2rgb,P1,P2,[],'Before Subpixel Fining');
cp_showMatch(I1rgb,I2rgb,P1,P3,[],'After Subpixel Fineing');
% �������������Ϣ
impixelinfo;