function x=symvec(X)
n=size(X,1);
x=zeros(0.5*n*(n+1),1);
for i=1:n
    x((n-0.5*i)*(i-1)+i:(n-0.5*(i+1))*i+i)=X(i:n,i);
end