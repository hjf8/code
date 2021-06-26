function Cholesky
A=[10 20 30;20 45 80;30 80 171];
%n=length(A);
n=2;
for j=1:n
    if j>1
        A(j:n,j)=A(j:n,j)-A(j:n,1:j-1)*A(j,1:j-1)';
    end
    A(j:n,j)=A(j:n,j)/sqrt(A(j,j));
end
A
C=tril(A)*tril(A)'