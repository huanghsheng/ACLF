clear;clc;close all;
%% 读取矩阵进行精细匹配得到校准后的两个匹配点
P1 = load('matrix1.mat');
P1=P1.matrix;
P2 = load('matrix2.mat');
P2=P2.matrix;
tic;
P3 = cp_subpixelFine(P1,P2); % Fine matching
%%
% 
%  PREFORMATTED
%  TEXT
% 
toc;
% P3=P2;
% 把P1和P3保存为matrix3,matrix4
path1 = 'matrix3.mat';
 
% 保存矩阵到.mat文件,matrix1.mat 和matrix3.mat 就是生成单应矩阵用的两个矩阵了
save(path1, 'P3');


%% 调用python脚本,误差也一起算了

system('E:\python-3.7.0\python.exe text3.py');




























