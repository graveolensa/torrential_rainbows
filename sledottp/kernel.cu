#include "kernel.h"
#define TX 32
#define TY 32

#define DIM 2100

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator-(const cuComplex& a) {
        return cuComplex(r-a.r, i-a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
    __device__ cuComplex operator/(const cuComplex& a) {
        return cuComplex((r*a.r + i*a.i)/(a.r*a.r + a.i*a.i), (i*a.r - r*a.i)/(a.r*a.r + a.i*a.i));
    }
};


__device__ cuComplex conj(cuComplex m)
{
    cuComplex out(m.r,-m.i);
    return out;
}


__device__ cuComplex nor(cuComplex m)
{
	cuComplex out(m.r*m.r+m.i*m.i,0.0);
	return out;
}

__device__ float  norg(cuComplex m)
{
    return sqrtf(m.r*m.r+m.i*m.i);
}




__device__ cuComplex qpoch(cuComplex a, cuComplex q) {
    cuComplex out(1.0,0.0);
    cuComplex unity(1.0,0.0);
    int i = 0;
    cuComplex Q = q;
    if(q.magnitude2()>1.0)
    {
        return cuComplex(0.0,0.0);
    }
    // We want to formally match the definition of a q-pochhammer symbol.
    for(i=1;i<80;i++)
    {
        out = out * (unity - a*Q);
        Q = q * Q;
    }
    return out;  
}

__device__ cuComplex qp(cuComplex a, cuComplex q, int n) {
    cuComplex out(1.0,0.0);
    cuComplex unity(1.0,0.0);
    int i = 0;
    cuComplex Q = q;
    if(q.magnitude2()>1.0)
    {
        return cuComplex(0.0,0.0);
    }
    // We want to formally match the definition of a q-pochhammer symbol.
    for(i=1;i<n;i++)
    {
        out = out * (unity - a*Q);
        Q = q * Q;
    }
    return out;
}

__device__ cuComplex ramphi(cuComplex q) {
    cuComplex out(1.0,0.0);
    cuComplex mone(-1.0,0.0);
    cuComplex mq = mone*q;
    return qpoch(mq,mq)/qpoch(q,mq);
}

__device__ cuComplex rampsi(cuComplex q) {
    cuComplex out(1.0,0.0);
    cuComplex mone(-1.0,0.0);
    cuComplex mq = mone*q;
    return qpoch(mq,q)*qpoch(q*q,q*q);
}

__device__ cuComplex ramchi(cuComplex q) {
    cuComplex out(1.0,0.0);
    cuComplex mone(-1.0,0.0);
    cuComplex mq = mone*q;
    return qpoch(mq,q*q);
}

__device__ cuComplex ramf(cuComplex a, cuComplex b) {
    cuComplex out(1.0,0.0);
    cuComplex mone(-1.0,0.0);
    cuComplex ma = mone*a;
    cuComplex mb = mone*b;
    return qpoch(ma,a*b)*qpoch(mb,a*b)*qpoch(a*b,a*b);
}






// complex exponential
__device__ cuComplex expc(cuComplex m)
{
  cuComplex out(expf(m.r) * cosf(m.i),expf(m.r) * sinf(m.i));
  return out;
}



__device__ cuComplex powc(cuComplex ag, cuComplex bg)
{  
  cuComplex out(0.0,0.0);
  cuComplex mesp(0.0,0.0);
  cuComplex frim(0.0,0.0);
  double radiu, thet;
  /* get the proper polar form of the complex number */
  radiu =  sqrtf(ag.r*ag.r + ag.i*ag.i);
  thet = atan2f(ag.i,ag.r);
  /* mesp gives R^(c+di) */
  mesp.r = powf(radiu,bg.r)*cosf(bg.i*logf(radiu));
  mesp.i = powf(radiu,bg.r)*sinf(bg.i*logf(radiu));
  /* frim gives e^(i theta (c+di)) */
  /* now since we already have the machinery
     for performing complex exponentiation (just exp), we
     can just call that here */
  frim.r = -1.0 * bg.i * thet;
  frim.i = bg.r * thet;
  frim = expc(frim);  
  out = mesp*frim;
  return out;
}


// cosine (nothing algorithmically clean)
__device__ cuComplex cosc(cuComplex m)
{
    cuComplex ai(0.0,1.0);
    cuComplex ot(0.5,0.0);
    cuComplex mone(-1.0,0.0);
    cuComplex out = ot*(expc(m*ai) + expc(mone*m*ai));
    return out;
}

__device__ cuComplex sins(cuComplex m)
{
    cuComplex ai(0.0,1.0);
    cuComplex ot(0.0,0.5);
    cuComplex mone(-1.0,0.0);
    cuComplex out = ot*(expc(m*ai) - expc(mone*m*ai));
    return out;
}

__device__ cuComplex tans(cuComplex m)
{
    return sins(m)/cosc(m);
}

__device__ cuComplex moeb(cuComplex t, cuComplex a, cuComplex z)
{
    cuComplex out(0.0,0.0);
    cuComplex ai(0.0,1.0);
    cuComplex unity(1.0,0.0);
    out = expc(ai*t) * (z-a)/(unity-conj(a)*z);
    return out;
}

__device__ cuComplex mob(cuComplex a, cuComplex z)
{
    cuComplex out(0.0,0.0);
    cuComplex ai(0.0,1.0);
    cuComplex unity(1.0,0.0);
    out =  (z-a)/(unity-conj(a)*z);
    return out;
}


__device__ cuComplex bnewt(cuComplex z) {
    cuComplex three(3.0,0.0);
    cuComplex unity(1.0,0.0);
    cuComplex out(0.0,0.0);
    cuComplex Z =z;
    cuComplex L(0.0,0.0);
    
    cuComplex R(0.62348980185873359,0.7818314824680298);
    cuComplex v(0.62348980185873359,0.7818314824680298);
    int i;
    for(i=0;i<100;i++)
    {
        L = sins(expc(Z)-cosc(Z))-Z;
        out = out + v*L;
        v = R * v;
        Z = Z - L/((expc(Z)+sins(Z))*cosc(expc(Z)-cosc(Z))-unity);
    }
    return out;
}

__device__ cuComplex they3(cuComplex z, cuComplex q)
{
    int u;
    cuComplex out(0.0,0.0);
    cuComplex enn(-20.0,0.0);
    cuComplex onn(1.0,0.0);
    cuComplex dui(0.0,1.0);
    for(u=-20;u<20;u++)
    {
        out = out + powc(q,enn*enn)*expc(dui*enn*z);
        enn = enn + onn;
    }
    return out;
}


__device__ cuComplex  wahi(cuComplex z)
{
    int u;
    cuComplex un(1.0,0.0);
    cuComplex ne(1.0,0.0);
 cuComplex out(0.0,0.0);
 for(u=1;u<40;u++)
 {
 	out = out + powc(z/ne,ne);
 	ne = ne + un;
 }
 out = out + un;
 return out;
}

__device__ cuComplex  dwahi(cuComplex z)
{
    int u;
    cuComplex un(1.0,0.0);
    cuComplex ne(1.0,0.0);
 cuComplex out(0.0,0.0);
 for(u=1;u<40;u++)
 {
 	out = out + powc(z/ne,ne-un);
 	ne = ne + un;
 }
 return out;
}


__device__ cuComplex they3p(cuComplex z, cuComplex q)
{
    int u;
    cuComplex out(0.0,0.0);
    cuComplex enn(-20.0,0.0);
    cuComplex onn(1.0,0.0);
    cuComplex dui(0.0,1.0);
    for(u=-20;u<20;u++)
    {
        out = out + (enn*enn)*powc(q,enn*enn-onn)*expc(dui*enn*z);
        enn = enn + onn;
    }
    return out;
}

__device__ cuComplex h3ey3p(cuComplex z, cuComplex q)
{
    int u;
    cuComplex out(0.0,0.0);
    cuComplex aut(0.0,0.0);
    cuComplex enn(-20.0,0.0);
    cuComplex onn(1.0,0.0);
    cuComplex dui(0.0,1.0);
    cuComplex vel(0.0,0.0);
    cuComplex rav(0.0,0.0);
    for(u=-40;u<40;u++)
    {
        vel = expc(dui*enn*z);
        rav = powc(q,enn*enn);
        aut = aut + (enn*enn)*rav/q*vel;
        out = out + rav*vel;
        enn = enn + onn;
    }
    return out/aut;
}


__device__ cuComplex thess(cuComplex z, cuComplex q)
{
	int v;
	cuComplex unity(1.0,0.0);
	cuComplex out(1.0,0.0);
	cuComplex tw(2.0,0.0);
	cuComplex qoo(1.0,0.0);
	 for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}






__device__ cuComplex the1(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     cuComplex rt(0.25,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return tw*out*powc(q,rt)*sins(z);
}

__device__ cuComplex the2(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     cuComplex rt(0.25,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity - tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return tw*out*powc(q,rt)*cosc(z);
}

__device__ cuComplex the3(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}


__device__ cuComplex the4(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity - tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}

/* routine to generate q-integers */
__device__ cuComplex qin(cuComplex a, cuComplex q)
{
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    out = (unity - powc(q, a))/(unity-q);
    return out;
}

/* generating function for n^2 */
__device__ cuComplex geffa(cuComplex z, cuComplex q)
{
    cuComplex out(0.0,0.0);
    cuComplex unity(1.0,0.0);
    cuComplex wu(0.0,0.0);
    cuComplex Z=unity;
    int v;
    for(v=0;v<20;v++)
    {
        out = out +  qin(wu*wu,q)* Z;
        wu = wu + unity; 
        Z = z * Z;
    }
return out;
}








__device__ cuComplex thratd(cuComplex z, cuComplex q)
{
	int n;
	cuComplex fau(4.0,0.0);
	cuComplex too(2.0,0.0);
	cuComplex unity(1.0,0.0);
	cuComplex ennn(1.0,0.0);
	cuComplex ni(-1.0,0.0);
	cuComplex noo(-1.0,0.0);
	cuComplex out(0.0,0.0);
	cuComplex loo = q;
	cuComplex qoo =q*q;
	for(n=0;n<80;n++)
	{
		out = out + noo*(loo/(unity-qoo))*sins(too*ennn*z);
		qoo = qoo * q*q;
		loo = loo * q;
		ennn = ennn +unity;
		noo = ni * noo;
	}
	return out*fau;
}

__device__ cuComplex thess4(cuComplex z, cuComplex q)
{
	int v;
	cuComplex unity(1.0,0.0);
	cuComplex out(1.0,0.0);
	cuComplex tw(2.0,0.0);
	cuComplex qoo(1.0,0.0);
	 for(v=0;v<20;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity - tw * qoo/q * cosc(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}




__device__ cuComplex thesk(cuComplex z, cuComplex q, cuComplex r)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
    cuComplex roo(1.0,0.0);
     for(v=0;v<20;v++)
    {
        qoo = qoo * q * q;
        roo = roo *  r * r ;
        out = out * (unity - qoo) * (unity + tw * qoo/q * cosc(tw*z) + roo*roo/(r*r)); 
        
    }
    return out;
}



__device__ cuComplex thass(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<20;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * sins(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}

__device__ cuComplex rogers( cuComplex q)
{
    cuComplex onf(0.2,0.0);
    cuComplex Q5 = q*q*q*q*q;
    cuComplex out = powc(q,onf)* qpoch(q,Q5) * qpoch(q*q*q*q,Q5)/ (qpoch(q*q,Q5)*qpoch(q*q*q,Q5));
    return out;
}

__device__ cuComplex flat(cuComplex m)
{
    float ua = sqrtf(m.r*m.r + m.i*m.i);
    cuComplex out(m.r/ua,m.i/ua);
    return out;
}

__device__ cuComplex eff(cuComplex z, cuComplex lambda)
{
    return z*z*z*z+ lambda/(z*z*z*z);
}

__device__  cuComplex thete(float R, cuComplex tau, cuComplex z)
{
    /* note that as I'm not immediately doing this on the unit circle, as the real
    action is considered to happen on the z-plane, we don't yet need to fret about
    whether I'm looking at things in terms of tau or in terms of q, next revision */
   /* set accumulant to zero */
    cuComplex A(0.0,0.0);
    /* miscellaneous setup */
    cuComplex pai(3.14159265353898,0.0);
    cuComplex ai(0.0,1.0);
    cuComplex oo(1.0,0.0);
    cuComplex oot(2.0,0.0);
    cuComplex nini(9.0,0.0);
    cuComplex eigh(-18.0,0.0);
    /* cuComplex arr(cos(2*3.1415926535897f*R/2048.0),0.0) */
    cuComplex frann(1.0,0.0);
    frann = pai * ai * tau ;
    cuComplex shenn(1.0,0.0);
    shenn = oot * ai * z;
    cuComplex plenn(1.0,0.0);
    cuComplex enn(1.0,0.0);
    cuComplex ann(1.0,0.0);
    cuComplex bnn(1.0,0.0);
    cuComplex scrunn(1.0,0.0);
    float ca, cb,cc;
    int a,  b;
    for(a=-10;a<10;a++)
    {
        ann.r = a;
        for(b=-10;b<10;b++)
        {
                bnn.r = b;
                if(((a+b)%2)==0)
                {
                        scrunn.r = a*a + b*b;
                        A = A + expc(frann* scrunn) * expc(shenn* (ann+bnn));
                }
                else
                {
                        
                        ca = 5.0 + a*a + b*b;
                        cb =  2*(a * cos(R)- b * sin(R));
                        cc  =  4*(b * cos(R)+a*sin(R));
                        scrunn.r = ca + cb + cc;
                        A = A + expc(frann*scrunn)*expc(shenn*(ann+bnn));
                }
        }
    }
    return A;
}
 
__device__  cuComplex thetta(cuComplex tau, cuComplex z)
{
    /* note that as I'm not immediately doing this on the unit circle, as the real
    action is considered to happen on the z-plane, we don't yet need to fret about
    whether I'm looking at things in terms of tau or in terms of q, next revision */
   /* set accumulant to zero */
    cuComplex A(0.0,0.0);
    /* miscellaneous setup */
    cuComplex pai(3.14159265353898,0.0);
    cuComplex ai(0.0,1.0);
    cuComplex oo(1.0,0.0);
    cuComplex oot(2.0,0.0);
    cuComplex nini(9.0,0.0);
    cuComplex eigh(-18.0,0.0);
    /* cuComplex arr(cos(2*3.1415926535897f*R/2048.0),0.0) */
    cuComplex frann(1.0,0.0);
    frann = pai * ai * tau ;
    cuComplex shenn(1.0,0.0);
    shenn = oot * ai * z;
    cuComplex plenn(1.0,0.0);
    cuComplex enn(1.0,0.0);
    int n;
    for(n=-10;n<10;n++)
    {
        enn.r = n;
        plenn = enn * enn;
        /* this get the cuComplex out of the event loop */
        A = A + expc(frann* plenn) * expc(shenn* enn);
}
return A;
}

__device__ cuComplex mitlef(cuComplex z,cuComplex c)
{
    cuComplex out(0.0,0.0);
    cuComplex Z(1.0,0.0);
    cuComplex frove(0.0,0.0);
    int v;
    for(v=0;v<20;v++)
    {
        frove.r = tgammaf(c.r*v+c.i);
        out = out + Z/frove;
        Z = Z * z;
    }
    return out;
}

__device__ cuComplex helva(cuComplex z)
{
    cuComplex out(j0f(z.r),j1f(z.i));
    return  out;
}

__device__ cuComplex hylva(cuComplex z)
{
    cuComplex out(j1f(1/j0f(z.r)),j1f(1/j0f(z.i)));
    return  out;
}


__device__ cuComplex hilva(cuComplex z)
{
    cuComplex out(j1f(z.r),j0f(z.i));
    return  out;
}


__device__ cuComplex halva(cuComplex z)
{
    cuComplex out(j0f(z.r),j0f(z.i));
    return  out;
}

__device__ cuComplex hinva(cuComplex z)
{
    cuComplex out(j1f(z.r),j1f(z.i));
    return  out;
}

__device__ cuComplex henga(cuComplex z)
{
    cuComplex out(acoshf(z.r),asinhf(z.i));
    return  out;
}

__device__ cuComplex holva(cuComplex z)
{
    cuComplex out(y0f(z.r),y1f(z.i));
    return  out;
}


__device__ cuComplex aliva(cuComplex z)
{
    cuComplex out(j1f(z.r),cyl_bessel_i1f(z.i));
    return  out;
}

__device__ cuComplex ariva(cuComplex z)
{
    cuComplex out(sinf(z.i),cbrtf(z.r));
    return  out;
}


__device__ cuComplex arago(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * hinva(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}


__device__ cuComplex irigo(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * holva(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}

__device__ cuComplex urigo(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * powc(hilva(q*z),helva(q*z)) + qoo*qoo/(q*q)); 
        
    }
    return out;
}


__device__ cuComplex arreg(cuComplex q, cuComplex r, cuComplex z)
{
	/* arreg implements the representation of theta3'(z)/theta(z) I don't know if these are
		derivatives with respect to z or q, we'll see */
	cuComplex out(0.0,0.0);
	cuComplex qoo(1.0,0.0);
	cuComplex roo(1.0,0.0);
	cuComplex morra(-1.0,0.0);
	cuComplex tla(1.0,0.0);
	cuComplex vnn(0.0,0.0);
	cuComplex fou(4.0,0.0);
	cuComplex tw(2.0,0.0);
	cuComplex run(1.0,0.0);
	int v;
	for(v=0;v<20;v++)
	{
		qoo = qoo * q;
		roo = roo * r * r;
		tla = tla * morra;
		vnn = vnn + run;
		out = out + morra*qoo*sins(tw*z*run)/(run-roo);
	}
return fou*out;
}

__device__ cuComplex urreg(cuComplex q, cuComplex r, cuComplex z)
{
	/* arreg implements the representation of theta3'(z)/theta(z) I don't know if these are
		derivatives with respect to z or q, we'll see */
	cuComplex out(0.0,0.0);
	cuComplex qoo(1.0,0.0);
	cuComplex roo(1.0,0.0);
	cuComplex morra(-1.0,0.0);
	cuComplex tla(1.0,0.0);
	cuComplex vnn(0.0,0.0);
	cuComplex fou(4.0,0.0);
	cuComplex tw(2.0,0.0);
	cuComplex run(1.0,0.0);
	int v;
	for(v=0;v<10;v++)
	{
		qoo = qoo * q;
		roo = roo * r * r;
		tla = tla * morra;
		vnn = vnn + run;
		out = out + morra*qoo*the3(tw*z*run,r)/(run-roo);
	}
return fou*out;
}


// * small q-exponential
__device__ cuComplex qexp(cuComplex z, cuComplex q)
{
	cuComplex mone(-1.0,0.0);
	cuComplex une(1.0,0.0);
	return une/qpoch(z,q);
}

//* large q exponential is just qpoch(-z,q)
__device__ cuComplex qExp(cuComplex z, cuComplex q)
{
	cuComplex mone(-1.0,0.0);
	cuComplex une(1.0,0.0);
	return qpoch(mone*z,q);
}




__device__ cuComplex sinq(cuComplex z, cuComplex q)
{
	cuComplex aie(0.0,1.0);
	cuComplex out(0.0,0.0);
	cuComplex doo(2.0,0.0);
	out = (qexp(z*aie,q) -qexp(z*aie,q))/doo;
	return out;
}


__device__ cuComplex cosq(cuComplex z, cuComplex q)
{
	cuComplex aie(0.0,1.0);
	cuComplex out(0.0,0.0);
	cuComplex doo(2.0,0.0);
	out = (qexp(z*aie,q) +qexp(z*aie,q))/doo;
	return out;
}

__device__ cuComplex Sinq(cuComplex z, cuComplex q)
{
	cuComplex aie(0.0,1.0);
	cuComplex out(0.0,0.0);
	cuComplex doo(2.0,0.0);
	out = (qExp(z*aie,q) -qExp(z*aie,q))/doo;
	return out;
}


__device__ cuComplex Cosq(cuComplex z, cuComplex q)
{
	cuComplex aie(0.0,1.0);
	cuComplex out(0.0,0.0);
	cuComplex doo(2.0,0.0);
	out = (qExp(z*aie,q) +qExp(z*aie,q))/doo;
	return out;
}

__device__ cuComplex asins(cuComplex z)
{
	float alp = 0.5 * (sqrtf((z.r+1)*(z.r+1) + z.i*z.i) + sqrtf((z.r-1)*(z.r-1) + z.i*z.i));
	float bet = 0.5 * (sqrtf((z.r+1)*(z.r+1) + z.i*z.i) - sqrtf((z.r-1)*(z.r-1) + z.i*z.i));
	float fla = z.i/abs(z.i);
	// *signum, but without a comparison, probably a saner way to do this? //
	cuComplex out(0.0,0.0);
	out.r = asinf(bet);
	out.i = fla * logf(alp + sqrtf(alp*alp-1));
	return out;
}


__device__ int gcd(int a, int b) 
{
    int remainder = a % b;
    if (remainder == 0) {
        return b;
    }
    return gcd(b, remainder);
}




/* Real Analytic Eisenstein Series */

__device__ cuComplex reis(cuComplex s, cuComplex z)
{
	// see en.wikipedia.org/wiki/Real_analytic_Eisenstein_series
	cuComplex out(0.0,0.0);
	cuComplex hav(0.5,0.0);
	cuComplex xu=out; 
	cuComplex yu=out;
	yu.r = z.i;
	int m,n;
	cuComplex ema=out;
	cuComplex ena=out;
	cuComplex den=out;
	for(m=-20;m<20;m++)
	{
		for(n=-20;n<20;n++)
		{
			if((m!=0)&&(n!=0))
			{
			if((gcd(m,n)==1))
			{
				ena.r = n;
				ema.r = m;
				den.r = norg(ema*z+ena);
				out = out + powc(yu,s)/powc(den,s/hav);
			}
			}
		}
	}
	return out;
}


__device__ cuComplex thu3(cuComplex z, cuComplex q)
{
    int v;
    cuComplex unity(1.0,0.0);
    cuComplex out(1.0,0.0);
    cuComplex tw(2.0,0.0);
    cuComplex qoo(1.0,0.0);
     for(v=0;v<10;v++)
    {
        qoo = qoo * q * q;
        out = out * (unity - qoo) * (unity + tw * qoo/q * asins(tw*z) + qoo*qoo/(q*q)); 
        
    }
    return out;
}

__device__ cuComplex trev(cuComplex lav, cuComplex mel, cuComplex rel)
{
cuComplex out(0.0,0.0);

cuComplex V(0.739085133215160641655312087674,0.0);
int v;
for(v=0;v<3;v++)
{
lav = lav - rel*(cosc(lav)-powc(V,rel))/cosc(lav);
out = out + mel*(cosc(lav)-powc(V,mel));
}

return out;
}

__device__ cuComplex polylog(cuComplex z, cuComplex s)
{
cuComplex out(0.0,0.0);
cuComplex oom(1.0,0.0);
cuComplex flag=oom;
int v;
for(v=0;v<30;v++)
{
    flag = flag + oom;
out = out + powc(z,flag)/powc(flag,s);
}

return out;
}

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r= blockIdx.y*blockDim.y + threadIdx.y;
  const int i = c + r*w; // 1D indexing
  float pi = 3.1415926535898;
  cuComplex ip(pi,0.0);
    const float scale = 
    4.3;
    float fx = scale * (float)(DIM/2 - c)/(DIM/2);
    float fy = scale * (float)(DIM/2 - r)/(DIM/2);
    cuComplex effx(fx,0.0);
    cuComplex effy(fy,0.0);
    float LA = scale * (float)(DIM/2 - pos.x)/(DIM/2);
    float LB = scale * (float)(DIM/2 - pos.y)/(DIM/2);
    cuComplex mouse(LA,LB);
    cuComplex moux(LA,0.0);
    cuComplex mouy(0.0,LB);
    cuComplex q(fx,fy);

/*    cuComplex tik(sin(ticks/40.0f),0.0);*/
/*    cuComplex uon(cosf(-2*pi*ticks/16384.0),sinf(-2*pi*ticks/16384.0));
    cuComplex aon(cosf(2.6457513110645912*2*pi*ticks/1024),sinf(2.645751311064591*2*pi*ticks/1024));
    cuComplex eon(cosf(-2.6457513110645912*2*pi*ticks/1024.0),sinf(2.645751311064591*2*pi*ticks/1024.0));*/
        cuComplex fixon(.029348,.828934);
    cuComplex faxon(.029348,-.828934);
    cuComplex unity(1.0,0.0);
    cuComplex ai(0.0,1.0);
    
   cuComplex  aon = expc(ai*moux);
   cuComplex uon= expc(mouy);

    cuComplex flurn(0.0,0.0);
    cuComplex accume(1.0,0.0);
    cuComplex eccume(1.0,0.0);
    cuComplex rhun(1.02871376821872462237195122725097462534904479,0.0);
    cuComplex cue = q;
    cuComplex lam(0.73736887807831963, -0.67549029426152396);
    cuComplex due(3.0,0.0);
    cuComplex tir(2.0,0.0);
    cuComplex selga(3.5,0.0);


    cuComplex vro(-1.0,0.0);
    cuComplex tle(0.0,0.0);
    cuComplex sle(4.0,0.0);
    cuComplex cherra(0.62348980185873359, 0.7818314824680298);
    cuComplex lerra = cherra*cherra;
    cuComplex ferra = lerra * cherra;
    cuComplex terra = ferra * cherra;
    cuComplex zerra = terra * cherra;
    cuComplex nerra = zerra * cherra;
cuComplex vlarv(1/3.0,0.0);
    cuComplex sugna(0.70710678118654757, 0.70710678118654746);
    cuComplex regna(0.99966573338968745, 0.025853848581176047);
    cuComplex spa(sqrtf(2.0),0.0);
    cuComplex spb(sqrtf(3.0),0.0);
    cuComplex spc(sqrtf(4.0),0.0);
    cuComplex spd(sqrtf(5.0),0.0);
    cuComplex mrun(1/2.0,0.0);
cuComplex gloon (4.0,0.0);
    cuComplex plenod(-.01,0.0);
cuComplex nue = cue;
cuComplex bor(-10.0,0.0);
cuComplex nat(0.0,-10.0);
cuComplex rhus(1.0,0.0);
cuComplex D(0.739085133215160641655312087674,0.0);
cuComplex gren(2.0,0.0);
cuComplex next=flurn;
cuComplex current = cue;
cuComplex xnext = flurn;
cuComplex xcurrent = cue;
cuComplex rue=cue;
cuComplex tinny(.0001,0.0001);
cuComplex raga(0.5,27.0);
cuComplex ruga(0.5,0.0);
cuComplex senna(2.0,0.0);
cuComplex renna(3.0,0.0);
cuComplex finch(0.001,.001);
cuComplex smenn(0.5,sqrtf(3.0)/2.0);
cuComplex lmenn(0.96592582628906831,0.25881904510252074);
cuComplex vmenn(-0.5,-sqrtf(3.0)/2.0);
float ah, ex, feig;
feig = 	3.67;
ex = 2.10981;
float xa,xb,ya,yb,tta,ttb;
char va,vb,vc;
cuComplex seahorse(-0.75,0.123);

/*  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int i = c + r*w; // 1D indexing
  const int dist = sqrtf((c - pos.x)*(c - pos.x) + 
                         (r - pos.y)*(r - pos.y));
  const unsigned char intensity = clip(255 - dist);*/
  
// theta function varying on constant
// cue =thess(cue,fixon*mouse);
int v=1;
int axa=-10;
int uu;


/*while((v<100)&&norg(cue)<2.0)
{
   cue = cue*(cue-mouy)*(cue-moux) -cue * q;
   v++;
}*/
   
// One way of describing this would be we want to perform Newton's method 
  //on the Mandelbrot set



/* preiterate */


//tex.stackexchange.com/questions/278843/making-a-phase-portrait-of-two-autonomous-system-of-differential-equations-with?fbclid=IwAR2Tz66CbUAq7LFVYck4uUGF5uQWnmzf5iZw3Bi8IOycvCC7czO6ZVgkz3s
// this is not terribly hard to do with cuda
   // what we need:
   // x' = x - y -> dx / dt = x - y
   // y' = 1 - x^2 -> dy / dt = 1-x^2
   // dy / dx = (dy / dt) / (dx/ dt)
   // so the trick is to convert dy/dx into a unit complex number to make this work, okay that's not that difficult

cue = mob(mouse,cue);
for(v=0;v<20;v++)
{
    cue = cue - mob(cue,mouse/mob(mouse,cue));
}









{

    double tha;
    tha = ((atan2(cue.i,cue.r) - pi)/(2.0*pi));

  
   d_out[i].x   = (unsigned char) (255.0*pow(sin(pi*tha),2));
    d_out[i].y = (unsigned char) (255.0*pow(sin(pi*tha+pi/3),2));
    d_out[i].z  = (unsigned char) (255.0*pow(sin(pi*tha+2*pi/3),2));
    d_out[i].w = 255;
 
   



    
}


}


 


void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  distanceKernel<<<gridSize, blockSize>>>(d_out, w, h, pos);
}

/*for(v=1;v<5;v++)
{
cue = cue - cue * (expc(unity-cue/moux)+expc(cue-unity/mouy))/((vlarv-unity/moux )*(expc(unity-cue/moux))-expc(cue-unity/mouy));
accume = accume + ((vlarv-unity/moux )*(expc(unity-cue/moux))-expc(cue-unity/mouy));
}

cue = accume;*/
/*cue = ramchi(moeb(unity,uon*fixon,q))*rampsi(moeb(unity,uon*fixon,q));
rhus = ramchi(uon/moeb(unity,uon*faxon,unity/q))*ramphi(uon/moeb(unity,uon*faxon,unity/q));
cue = rhus+cue;
cue = cosc(unity/(unity-uon*cue))*rampsi(moeb(unity,uon*fixon,q));*/


/*for(v=0;v<60;v++){
        cue = moeb(aon,fixon,cue) - aon/((expc(uon*cue-sins(cue))-cue)/((aon+cosc(cue)) * expc(uon*cue-sins(cue))-aon));
        accume = accume *(unity - (expc(aon*moeb(uon,faxon,cue))-sins(moeb(aon,fixon,cue))-cue));
    }
    cue = accume;*/

/*
One for
(x+d)/cos(d) -cos(x)/d
Tungilipa

D = cos(D)

cos(sqrt(x*D))/D -1 = 0.0




The other for
cos(x)-x
Eripgrunna
*/