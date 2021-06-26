function [err_global_H1,err_global_L2] = main(N)
%% ��ά����Ԫ����
tic;
clc;  
clear; 
close all; 
L_x = 1;%x�������䳤�� 
L_y = 1;%y�������䳤�� 
N = 4;
n_x = N;%x����ľ��θ���
n_y = N;%y����ľ��θ���
h_x = L_x/n_x;%x�����ϵĵ�Ԫ����
h_y = L_y/n_y;%y�����ϵĵ�Ԫ����
num = 2*n_x*n_y;%�����εĸ���
u_b = zeros(2*(n_x+n_y),1); %�߽�����
node_x = n_x + 1; % x����ڵ����
node_y = n_y + 1; % y����ڵ����
node_sum = node_x*node_y;%�ܽڵ����
nel = 3;%���ɶ�
x = linspace(0,L_x,node_x)'; %�ȷֽڵ�ĺ�����
y = linspace(0,L_y,node_y)'; %�ȷֽڵ��������
[X, Y] = meshgrid(x,y);%�ų�����X��Y�ֱ��ʾ��Ӧλ�õĺ�������
X = X';Y = Y';coord = [X(:) Y(:)];%ÿһ���Ƕ�Ӧ�ڵ������
lianjie_mat = lianjie(node_x,node_y,nel);%���Ӿ��󣬱�ʾÿ����Ԫ��Χ�Ľڵ��ţ�Ҳ�����漰���κ������
border = unique([1:node_x node_x*n_y+1:node_x*n_y+node_x ...%�������ߵı߽�
    node_x+1:node_x:node_x*(n_y-1)+1 2*node_x:node_x:n_y*node_x]);%�������� 
%ebcval = u_b; %����߽�ֵ��Ϊu_b
K = sparse(node_sum,node_sum); % �նȾ���[K]����ʼ��Ϊ0��
F = sparse(node_sum,1);      % �Ҷ���,��ʼ��Ϊ0
%% �������ܸ�
%  ����ϵ������K���Ҷ���f
for i = 1:num %ͬһά���������Ȼ����Ԫ��ɨ��
  k = xishu(i,nel,h_x,h_y,coord,lianjie_mat);%���㵥Ԫ�նȾ���
  f = youduan(i,nel,h_x,h_y,coord,lianjie_mat);%���㵥Ԫ�غ�����
  m = lianjie_mat(i,:);
  K(m,m) = K(m,m) + k;
  F(m) = F(m) + f;
  %full(K)
  %full(F)
end
%full(K);
%% ��ֵ��������
for i = 1:length(border)
    n = border(i);
    for j = 1:node_sum
        if (isempty(find(border == j, 1))) % ��j���������ǹ̶���
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
%% ��������
e = sparse(num,2);%��һ�д洢H1��ģ�ֲ����ڶ��д洢L2�ֲ����
for i = 1:num
    [e(i,1),e(i,2)] = err_local(i,u,h_x,h_y,coord,lianjie_mat); 
    %e(i,1) = err_H1;  
    %e(i,2) = err_L2;
end
err_global_H1 = sqrt(sum(e(:,1)))%H1��ģ
err_global_L2 = sqrt(sum(e(:,2)))%L2ģ
t=toc

%{
u_re = reshape(u,node_x,node_y);
u_re = full(u_re);
%figure
%mesh(x,y,u_re);
%title('Approximate Solutions');
% ��ȷ��
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






