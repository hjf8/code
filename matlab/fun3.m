function h=fun3(a,b,c)
d=zeros(2);
h=zeros(2);
for i=1:2
    for j=1:2
        for k=1:2
        d(i,j)=d(i,j)+a(i,k)*b(k,j);
        end
    end
end
for m=1:2
    for n=1:2
        for l=1:2
            h(m,n)=h(m,n)+d(m,l)*c(l,n);
        end 
    end
end
disp(h);
