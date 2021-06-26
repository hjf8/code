clear;
clc
close all;
Err1 = [];
Err2 = [];
for i=[10,20,100]
    [err_global_H1,err_global_L2] = main(i);  
    Err1(end+1) = err_global_H1;
    Err2(end+1) = err_global_L2;
end
Err1,Err2,
%{
r1=[];
r2=[];
for i=1:N-1
    r1(i)=(log2(Err1(i)/Err1(i+1)))/(log2((i+1)/i));
    r2(i)=(log2(Err2(i)/Err2(i+1)))/(log2((i+1)/i));
end
r1,r2
%}