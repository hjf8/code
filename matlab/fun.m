function c=fun(n,k)
tic;
x=round(100*rand(n,1));
y=round(100*rand(n,1));
c=y'*x; 
c=c^(k-1)*(x*y');
%disp(c);
t=toc
end

 