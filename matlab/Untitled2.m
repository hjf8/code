
n=3;
A.vec = symvec(A)
B.vec=zeros(size(A.vec));
for i=1:n
    for j=i:n
        XX=X(:,i)*X(:,j)';
        B.vec((n-0.5*i)*(i-1)+j)=XX'*A.vec;
    end
end