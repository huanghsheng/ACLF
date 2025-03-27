function P2fine = cp_subpixelFine(P1,P2)
%CP_SUBPIXELFINE refine the coarse match coordinates in subpixel level
len = length(P1);
affmat = fitgeotrans(P1,P2,'projective'); % 创建透视变换矩阵
P2pro = [P1 ones(len,1)] * affmat.T;
P2pro = P2pro(:,1:2) ./ [P2pro(:,3) P2pro(:,3)];
devia_P = (P2 - P2pro).^2;
devia_P = sqrt(sum(devia_P,2));
max_Devia = max( devia_P );
iteration = 0;
P2fine = P2;
while max_Devia > 0.05 && iteration < 20
    iteration = iteration+1;
    fprintf('\nsubpixel iteration = %d\tmax_devia=%f\n',iteration,max_Devia);
    [~,index] = sort(devia_P); % devia_P表示残差，残差越大，表示越有可能是拟合点，按升序排序
    ind1 = round( 1/4 * length(index));
    P2fine(index(ind1:end),:)    =  P2pro(index(ind1:end),:); % 取P2pro拟合点矩阵后3/4 的部分替换P2fine,P2fine前1/4 不变
    affmat = fitgeotrans(P1,P2fine,'projective');
    P2pro = [P1 ones(len,1)] * affmat.T;
    P2pro = P2pro(:,1:2) ./ [P2pro(:,3) P2pro(:,3)]; 
    devia_P = (P2fine - P2pro).^2;  % 最终点减去P2pro的点，然后平方
    devia_P = sqrt(sum(devia_P,2));  % 每行相加，然后取根号
    max_Devia = max( devia_P );  % 取一列中的最大值
end
fprintf('\nsubpixel iteration = %d\tmax_devia=%f\n',iteration,max_Devia);

