function [err_H1,err_L2]=err_local(i,u,h_x,h_y,coord,lianjie)
syms x y
S = h_x*h_y/2;%三角形面积
nodes = lianjie(i,:);%相关形函数（节点）编号
xe = coord(nodes,:);%相关节点的坐标
ui = u(nodes);
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);xi3 = xe(1,1) - xe(2,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);eta3 = xe(1,2) - xe(2,2);
w1 = xe(2,1)*xe(3,2)-xe(3,1)*xe(2,2);w2=xe(3,1)*xe(1,2)-xe(1,1)*xe(3,2);w3=xe(1,1)*xe(2,2)-xe(2,1)*xe(1,2);
%基函数
lamd1 = (eta1*x-xi1*y+w1)/(2*S);
lamd2 = (eta2*x-xi2*y+w2)/(2*S);
lamd3 = (eta3*x-xi3*y+w3)/(2*S);
u_local = ui(1,1)*lamd1+ui(2,1)*lamd2+ui(3,1)*lamd3;
u_grad = gradient(u_local,[x ,y])-gradient(sin(pi*x).*sin(pi*y),[x ,y]);
u_ex = sin(pi*x).*sin(pi*y);%精确解
u_in1 = sum((u_grad).^2);%H1半模
u_in2 = sum((u_local - u_ex).^2);%L2模
x_average = (xe(1,1)+xe(2,1)+xe(3,1))/3;y_average=(xe(1,2)+xe(2,2)+xe(3,2))/3;
err_H1 = vpa(S*(subs(subs(u_in1,x,x_average),y,y_average)),10);
err_L2 = vpa(S*(subs(subs(u_in2,x,x_average),y,y_average)),10);



