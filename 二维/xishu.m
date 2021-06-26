function [k] = xishu(i,nel,h_x,h_y,coord,lianjie)
S = h_x*h_y/2;%三角形面积
k = zeros(nel,nel); 
nodes = lianjie(i,:);%相关形函数（节点）编号
x    = coord(nodes,:);%相关节点的坐标
xi1 = x(2,1) - x(3,1);xi2 = x(3,1) - x(1,1);%xi3 = x(1,1) - x(2,1);
eta1 = x(2,2) - x(3,2);eta2 = x(3,2) - x(1,2);%eta3 = x(1,2) - x(2,2);
auv11 = -((eta1*xi2 - eta2*xi1)*(eta1^2 + eta2^2))/(8*S^2);
auv12 = ((eta1*xi1 + eta2*xi2)*(eta1*xi2 - eta2*xi1))/(8*S^2);
auv13 = -((eta1*xi2 - eta2*xi1)*(- eta1^2 + xi1*eta1 - eta2^2 + xi2*eta2))/(8*S^2);
auv22 = -((xi1^2 + xi2^2)*(eta1*xi2 - eta2*xi1))/(8*S^2);
auv23 = -((eta1*xi2 - eta2*xi1)*(- xi1^2 + eta1*xi1 - xi2^2 + eta2*xi2))/(8*S^2);
auv33 = -((eta1*xi2 - eta2*xi1)*(eta1^2 - 2*eta1*xi1 + eta2^2 - 2*eta2*xi2 + xi1^2 + xi2^2))/(8*S^2);
k = k + [auv11 auv12 auv13;auv12 auv22 auv23;auv13 auv23 auv33];
return
end