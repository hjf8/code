function m=fun6(x,y,k)
x=round(100*rand(2,1));
y=round(100*rand(2,1));
c=x*y';
m=c^k;
disp(m);