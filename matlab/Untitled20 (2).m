function c=fun_c(f,b,lamuda,n) %fΪ�Ҷ���
B=zeros(n);
for i=1:n
    for j=1:n
        B(i,j)=B(i,j)-b^(i+j-1)/(factorial(j-1)*(i+j-1));
    end
end
B=lamuda*eye(n)+B;
c=B\f;