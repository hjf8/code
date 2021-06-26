function [GL,Y,RGn]=GaussR1(fun,X,A)
n=length(X);n2=2*n;Y=feval(fun,X);GL=sum(A.*Y);
sun=1;su2n=1;su2n1=1;wome=1;
syms x
for k=1:n
    wome=wome*(x-X(k));
end
wome2=wome^2;Fr=int(wome2,x,-1,1);
for k=1:n2
    su2n=su2n*k;
end
syms M
RGn=Fr*M/su2n;