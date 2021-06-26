function [x]=Block(S,T,lamuda,b)
n=length(S);
if n<=1
    x=b(n)/(S(n,n)*T(n,n)-lamuda);
else
    [x]=Block(S(2:n,2:n),T(2:n,2:n),lamuda,b(2:n));
    w=T(2:n,2:n)*x;
    r=(b(1)-S(1,1)*(T(1,2:n)*x)-S(1,2:n)*w)/(S(1,1)*T(1,1)-lamuda);
    x=[r;x];
end