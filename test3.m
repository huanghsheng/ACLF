clear;clc;close all;
%% ��ȡ������о�ϸƥ��õ�У׼�������ƥ���
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
% ��P1��P3����Ϊmatrix3,matrix4
path1 = 'matrix3.mat';
 
% �������.mat�ļ�,matrix1.mat ��matrix3.mat �������ɵ�Ӧ�����õ�����������
save(path1, 'P3');


%% ����python�ű�,���Ҳһ������

system('E:\python-3.7.0\python.exe text3.py');




























