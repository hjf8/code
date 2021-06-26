function [f] = youduan(i,nel,h_x,h_y,coord,lianjie)
nodes = lianjie(i,:);%自由度编号
xe = coord(nodes,:); % 单元自由节点坐标
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);%xi3 = xe(1,1) - xe(2,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);%eta3 = xe(1,2) - xe(2,2);
detJ = eta2*xi1 - eta1*xi2;
g1 = @(lam1,lam2) fun(xe(3,1) + lam1*(xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*lam1*detJ;
g2 = @(lam1,lam2) fun(xe(3,1) + lam1*(xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*lam2*detJ;
g3 = @(lam1,lam2) fun(xe(3,1) + lam1*(xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*(1-lam1-lam2)*detJ;

lammax = @(lam1) 1 - lam1;
gx(1) = integral2(g1,0,1,0,lammax);
gx(2) = integral2(g2,0,1,0,lammax);
gx(3) = integral2(g3,0,1,0,lammax);
f = [gx(1);gx(2);gx(3)];
end
