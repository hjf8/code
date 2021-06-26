function d=fun4(m,n,p,q)
a=round(100*rand(m,n));
b=round(100*rand(n,p));
c=round(100*rand(p,q));
tic;
d=a*b*c;
%disp(d);
t1=100*toc
tic;
d=a*(b*c);
%disp(e);
t2=100*toc