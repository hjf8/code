x=[-5:1:5];
y=1./(1+x.^2);
x0=[-5:0.001:5];
y0=Lagrange(x,y,x0);
y1=1./(1+x0.^2);
plot(x0,y0,'b')
hold on
plot(x0,y1,'r')
