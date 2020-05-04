#include <stdio.h>
#include <stdlib.h>

/* parameters of Mersenne Twister */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

/* parameters for computing Jump */
#define W_SIZE 32 /* size of unsigned long */
#define MEXP 19937
#define P_SIZE ((MEXP / W_SIZE) + 1)
#define LSB 0x00000001ul
#define QQ 7
#define LL 128 /* LL = 2^(QQ) */

typedef struct
{
    unsigned long s[N];
    int ptr;
} State;

/* operations for polynomials */
void random_poly(unsigned long *);
void gray_code(void);
unsigned long get_coef(unsigned long *, unsigned int);
void set_coef(unsigned long *, unsigned int, unsigned long);

/* operations for states */
State *calc_state(unsigned long *, State *);
void copy_state(State *, State *);
State *horner1(unsigned long *, State *);
void gen_next(State *);
void add_state(State *, State *);
int compare_state(State *, State *);
void gen_vec_h(State *);

/* initial state */
State s0;

unsigned long h[LL];
State vec_h[LL];

int main(void)
{
    unsigned long *pf;
    State *ss1, *ss2, ss3;
    int i, j, deg;
    char c;
    FILE *fin, *file;

    pf = (unsigned long *)calloc(P_SIZE, sizeof(unsigned long));

    /* read the file state.txt, and set the state */
    file = fopen("state.txt", "r");

    if (file == NULL)
    {
        printf("Error Reading File\n");
        return -1;
    }

    for (i = 0; i < N; i++)
    {
        fscanf(file, "%lu", &s0.s[i]);
        printf("s0.s[%d]: %lu \n", i, s0.s[i]);
    }
    fscanf(file, "%d", &s0.ptr);
    printf("s0.ptr: %d \n", s0.ptr);
    printf("========== \n");

    /* read the file clist.txt, and set the coefficient */
    if ((fin = fopen("jump-poly.txt", "r")) == NULL)
    {
        printf("File read error.\n");
        exit(1);
    }

    for (i = MEXP - 1; i > -1; i--)
    {
        c = fgetc(fin);
        if (c == '1')
            set_coef(pf, i, 1);
    }

    for (i = 0; i < P_SIZE; i++)
    {
        printf("pf[%d]: %lu\n", i, pf[i]);
    }

    /* computes jumping ahead with standard Horner method */
    ss1 = horner1(pf, &s0);

    /* computes jumping ahead with Sliding window algorithm */
    gen_vec_h(&s0);
    ss2 = calc_state(pf, &s0);

    for (i = 0; i < N; i++)
    {
        printf("s0.s[%d]: %lu \n", i, s0.s[i]);
    }
    printf("s0.ptr: %d \n", s0.ptr);
    printf("========== \n");

    for (i = 0; i < N; i++)
    {
        printf("ss1.s[%d]: %lu \n", i, ss1->s[i]);
    }
    printf("ss1.ptr: %d \n", ss1->ptr);
    printf("========== \n");

    if (compare_state(ss1, ss2) != 0)
        printf("error\n");
    return (1);

    return (0);
}

/* generate random polynomial */
void random_poly(unsigned long *p)
{
    int i;

    for (i = 0; i < MEXP; i++)
        set_coef(p, i, rand() % 2);
}

/* copy state ss */
void copy_state(State *ts, State *ss)
{
    int i;

    for (i = 0; i < N; i++)
        ts->s[i] = ss->s[i];

    ts->ptr = ss->ptr;
}

/* compute pf(ss) using standard Horner method */
State *horner1(unsigned long *pf, State *ss)
{
    int i = MEXP - 1;
    State *temp;

    temp = (State *)calloc(1, sizeof(State));

    while (get_coef(pf, i) == 0)
        i--;

    if (i > 0)
    {
        copy_state(temp, ss);
        gen_next(temp);
        i--;
        for (; i > 0; i--)
        {
            if (get_coef(pf, i) != 0)
                add_state(temp, ss);
            else
                ;
            gen_next(temp);
        }
        if (get_coef(pf, 0) != 0)
            add_state(temp, ss);
        else
            ;
    }
    else if (i == 0)
        copy_state(temp, ss);
    else
        ;

    return (temp);
}

/* compute next state */

/* add s2 to s1 */
void add_state(State *s1, State *s2)
{
    int i, pt1 = s1->ptr, pt2 = s2->ptr;

    if (pt2 - pt1 >= 0)
    {
        for (i = 0; i < N - pt2; i++)
            s1->s[i + pt1] ^= s2->s[i + pt2];
        for (; i < N - pt1; i++)
            s1->s[i + pt1] ^= s2->s[i + (pt2 - N)];
        for (; i < N; i++)
            s1->s[i + (pt1 - N)] ^= s2->s[i + (pt2 - N)];
    }
    else
    {
        for (i = 0; i < N - pt1; i++)
            s1->s[i + pt1] ^= s2->s[i + pt2];
        for (; i < N - pt2; i++)
            s1->s[i + (pt1 - N)] ^= s2->s[i + pt2];
        for (; i < N; i++)
            s1->s[i + (pt1 - N)] ^= s2->s[i + (pt2 - N)];
    }
}

/* compute pf(ss) using Sliding window algorithm */
State *calc_state(unsigned long *pf, State *ss)
{
    State *temp1;
    int i = MEXP - 1, j, digit, skip = 0;

    temp1 = (State *)calloc(1, sizeof(State));

    while (get_coef(pf, i) == 0)
        i--;

    for (; i >= QQ; i--)
    {
        if (get_coef(pf, i) != 0)
        {
            for (j = 0; j < QQ + 1; j++)
                gen_next(temp1);
            digit = 0;
            for (j = 0; j < QQ; j++)
                digit = (digit << 1) ^ get_coef(pf, i - j - 1);
            add_state(temp1, &vec_h[digit]);
            i -= QQ;
        }
        else
            gen_next(temp1);
    }

    for (; i > -1; i--)
    {
        gen_next(temp1);
        if (get_coef(pf, i) == 1)
            add_state(temp1, ss);
        else
            ;
    }
    return (temp1);
}

/* compare s1 with s2 */
/* if s1 is different from s2, then return 1 , else return 0 */
int compare_state(State *s1, State *s2)
{
    int i, d = 0;

    if ((s1->s[s1->ptr] & 0x80000000ul) != (s2->s[s2->ptr] & 0x80000000ul))
        d = 1;

    for (i = 1; i < N; i++)
        if (s1->s[(i + s1->ptr) % N] != s2->s[(i + s2->ptr) % N])
        {
            d = 1;
            break;
        }

    return (d);
}

/* generate Gray code */
void gray_code(void)
{
    unsigned int i, j = 1, l = 1, term = LL;

    h[0] = 0;

    for (i = 1; i <= QQ; i++)
    {
        l = (l << 1);
        term = (term >> 1);
        for (; j < l; j++)
            h[j] = h[l - j - 1] ^ term;
    }
}

/* compute h(f)ss where h(t) are exact q-degree polynomials, */
/* f is the tarnsition function, and ss is the initial state */
/* the results are stored in vec_h[0] , ... , vec_h[LL-1]    */
void gen_vec_h(State *ss)
{
    int i;
    unsigned long k, g;
    State v;

    gray_code();

    copy_state(&vec_h[0], ss);

    for (i = 0; i < QQ; i++)
        gen_next(&vec_h[0]);

    for (i = 1; i < LL; i++)
    {
        copy_state(&v, ss);
        g = h[i] ^ h[i - 1];
        for (k = 1; k < g; k = (k << 1))
            gen_next(&v);
        copy_state(&vec_h[h[i]], &vec_h[h[i - 1]]);
        add_state(&vec_h[h[i]], &v);
    }
}

/* 32-bits function */
/* return the i-th coefficient of the polynomial pf */
unsigned long get_coef(unsigned long *pf, unsigned int deg)
{
    if ((pf[deg >> 5] & (LSB << (deg & 0x1ful))) != 0)
        return (1);
    else
        return (0);
}

/* 32-bit function */
/* set the coefficient of the polynomial pf with v */
void set_coef(unsigned long *pf, unsigned int deg, unsigned long v)
{
    if (v != 0)
        pf[deg >> 5] ^= (LSB << (deg & 0x1ful));
    else
        ;
}

/* next state generating function */
void gen_next(State *ss)
{
    int num;
    unsigned long y;
    static unsigned long mag02[2] = {0x0ul, MATRIX_A};

    num = ss->ptr;
    if (num < N - M)
    {
        y = (ss->s[num] & UPPER_MASK) | (ss->s[num + 1] & LOWER_MASK);
        ss->s[num] = ss->s[num + M] ^ (y >> 1) ^ mag02[y % 2];
        ss->ptr++;
    }
    else if (num < N - 1)
    {
        y = (ss->s[num] & UPPER_MASK) | (ss->s[num + 1] & LOWER_MASK);
        ss->s[num] = ss->s[num + (M - N)] ^ (y >> 1) ^ mag02[y % 2];
        ss->ptr++;
    }
    else if (num == N - 1)
    {
        y = (ss->s[N - 1] & UPPER_MASK) | (ss->s[0] & LOWER_MASK);
        ss->s[N - 1] = ss->s[M - 1] ^ (y >> 1) ^ mag02[y % 2];
        ss->ptr = 0;
    }
}
