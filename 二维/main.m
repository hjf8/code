function [err_global_H1,err_global_L2] = main(N)
%% 二维有限元程序
tic;
clc;  
clear; 
close all; 
L_x = 1;%x方向区间长度 
L_y = 1;%y方向区间长度 
N = 4;
n_x = N;%x方向的矩形个数
n_y = N;%y方向的矩形个数
h_x = L_x/n_x;%x方向上的单元长度
h_y = L_y/n_y;%y方向上的单元长度
num = 2*n_x*n_y;%三角形的个数
u_b = zeros(2*(n_x+n_y),1); %边界条件
node_x = n_x + 1; % x方向节点个数
node_y = n_y + 1; % y方向节点个数
node_sum = node_x*node_y;%总节点个数
nel = 3;%自由度
x = linspace(0,L_x,node_x)'; %等分节点的横坐标
y = linspace(0,L_y,node_y)'; %等分节点的纵坐标
[X, Y] = meshgrid(x,y);%张成网格，X和Y分别表示对应位置的横纵坐标
X = X';Y = Y';coord = [X(:) Y(:)];%每一行是对应节点的坐标
lianjie_mat = lianjie(node_x,node_y,nel);%连接矩阵，表示每个单元周围的节点编号，也就是涉及的形函数编号
border = unique([1:node_x node_x*n_y+1:node_x*n_y+node_x ...%上下两边的边界
    node_x+1:node_x:node_x*(n_y-1)+1 2*node_x:node_x:n_y*node_x]);%左右两边 
%ebcval = u_b; %假设边界值都为u_b
K = sparse(node_sum,node_sum); % 刚度矩阵[K]，初始化为0，
F = sparse(node_sum,1);      % 右端项,初始化为0
%% 单刚组总纲
%  计算系数矩阵K和右端项f
for i = 1:num %同一维的情况，依然按单元来扫描
  k = xishu(i,nel,h_x,h_y,coord,lianjie_mat);%计算单元刚度矩阵
  f = youduan(i,nel,h_x,h_y,coord,lianjie_mat);%计算单元载荷向量
  m = lianjie_mat(i,:);
  K(m,m) = K(m,m) + k;
  F(m) = F(m) + f;
  %full(K)
  %full(F)
end
%full(K);
%% 边值条件处理
for i = 1:length(border)
    n = border(i);
    for j = 1:node_sum
        if (isempty(find(border == j, 1))) % 第j个点若不是固定点
            F(j) = F(j) - K(j,n)*u_b(i);
        end
    end
    K(n,:) = 0.0;
    K(:,n) = 0.0;
    K(n,n) = 1.0;
    F(n) = u_b(i);
end
%full(F);
%full(K);
u = K\F;
%full(u);
%% 计算误差阶
e = sparse(num,2);%第一列存储H1半模局部误差，第二列存储L2局部误差
for i = 1:num
    [e(i,1),e(i,2)] = err_local(i,u,h_x,h_y,coord,lianjie_mat); 
    %e(i,1) = err_H1;  
    %e(i,2) = err_L2;
end
err_global_H1 = sqrt(sum(e(:,1)))%H1半模
err_global_L2 = sqrt(sum(e(:,2)))%L2模
t=toc

%{
u_re = reshape(u,node_x,node_y);
u_re = full(u_re);
%figure
%mesh(x,y,u_re);
%title('Approximate Solutions');
% 求精确解
L =L_x;
n_exact = N;
x_exact = linspace(0,L,n_exact);
[X1,Y1] = meshgrid(x_exact,x_exact);
u_exact = exactsolution(X1(:),Y1(:));
%u_exact_re = reshape(u_exact,n_exact,n_exact);
%figure
%mesh(x_exact,x_exact,u_exact_re);
%title(' Exact Solutions');
u_ex = exactsolution(coord(:,1),coord(:,2));
error_L2 = sqrt(sum((u - u_ex).^2)*h_x*h_y);
%}






