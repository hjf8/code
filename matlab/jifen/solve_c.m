function c=solve_c(f,b,lambda,n) %fÎªÓÒ¶ËÏî
B=ones(n,n);
for i=1:n
    for j=1:n
        B(i,j)=-b^(i+j-1)/(factorial(j-1)*(i+j-1));
    end
    B(i,i)=lambda+B(i,i);
end
c=B\f;
end