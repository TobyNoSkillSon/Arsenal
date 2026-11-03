/* SPDX-License-Identifier: MIT */
#include "json.h"
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <mach/mach.h>

#define MAX_FRAME (16u*1024u*1024u)
static pthread_mutex_t output_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t state_lock=PTHREAD_MUTEX_INITIALIZER;
static atomic_int cancelled;
static int active_id=0, loaded=0, joinable=0;
static pthread_t thread;
static double scale=1;
typedef struct { int id, count; double values[64], scale; uint64_t iterations; } Job;

static int exact(void *buffer,size_t n,int eof) {
    size_t used=0;
    while(used<n) { size_t read=fread((char*)buffer+used,1,n-used,stdin); if(!read)return used==0 && eof ? 0 : -1; used+=read; }
    return 1;
}
static void send_json(const char *json) {
    size_t n=strlen(json)+1;
    if(n>MAX_FRAME)exit(2);
    unsigned char header[5]={(unsigned char)(n>>24),(unsigned char)(n>>16),(unsigned char)(n>>8),(unsigned char)n,1};
    pthread_mutex_lock(&output_lock);
    if(fwrite(header,1,5,stdout)!=5 || fwrite(json,1,n-1,stdout)!=n-1 || fflush(stdout))exit(2);
    pthread_mutex_unlock(&output_lock);
}
static void status(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&info,&count)!=KERN_SUCCESS)exit(2);
    char json[256];
    snprintf(json,sizeof(json),"{\"event\":\"status\",\"memory\":{\"footprint_mb\":%.6f,\"mlx_active_mb\":0,\"mlx_cache_mb\":0}}",(double)info.resident_size/1000000);
    send_json(json);
}
static void reply(int id,const char *fields) {
    char json[512]; status(); snprintf(json,sizeof(json),"{\"id\":%d,\"ok\":true%s}",id,fields?fields:""); send_json(json);
}
static void error(int id,const char *code,const char *message) {
    char json[512]; snprintf(json,sizeof(json),"{\"id\":%d,\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}",id,code,message); send_json(json);
}
static void *evaluate(void *pointer) {
    Job *job=pointer; double result=0;
    for(uint64_t i=0;i<job->iterations;i++) {
        if((i%1024)==0 && atomic_load(&cancelled))goto cancel;
        result=0; for(int k=0;k<job->count;k++)result+=job->values[k]; result*=job->scale;
    }
    if(atomic_load(&cancelled))goto cancel;
    { char fields[96]; snprintf(fields,sizeof(fields),",\"result\":%.17g",result); reply(job->id,fields); }
    goto done;
cancel: error(job->id,"cancelled","Request cancelled.");
done:
    pthread_mutex_lock(&state_lock); active_id=0; pthread_mutex_unlock(&state_lock); free(job); return NULL;
}
static void finish_job(void) {
    atomic_store(&cancelled,1);
    if(joinable) { pthread_join(thread,NULL); joinable=0; }
}
static int request(JSON *j) {
    double value;
    if(!number(j,field(j,0,"id"),&value) || value<1 || value>2147483647 || floor(value)!=value)return -1;
    int id=(int)value, op=field(j,0,"op"); if(op<0 || j->t[op].type!=JSTRING)return -1;
    if(text_eq(j,op,"cancel")) {
        double target=0; number(j,field(j,0,"target"),&target);
        pthread_mutex_lock(&state_lock); if(target==active_id)atomic_store(&cancelled,1); pthread_mutex_unlock(&state_lock);
        char json[80]; snprintf(json,sizeof(json),"{\"id\":%d,\"ok\":true}",id); send_json(json);
    } else if(text_eq(j,op,"load")) {
        pthread_mutex_lock(&state_lock); int busy=active_id!=0; pthread_mutex_unlock(&state_lock);
        if(busy) { error(id,"busy","Runner is busy."); return 0; }
        int config=field(j,0,"config"), settings=field(j,config,"settings"), factor=field(j,settings,"scale");
        if(factor<0)factor=field(j,config,"scale");
        double n=1;
        if(factor>=0 && (!number(j,factor,&n) || fabs(n)>100)) { error(id,"invalid_request","Scale must be a number from -100 to 100."); return 0; }
        scale=n; loaded=1; reply(id,",\"load_s\":0,\"engine\":\"cpu-toy\"");
    } else if(text_eq(j,op,"ping") || text_eq(j,op,"status"))reply(id,NULL);
    else if(text_eq(j,op,"unload")) { finish_job(); loaded=0; reply(id,NULL); return 1; }
    else if(text_eq(j,op,"evaluate")) {
        int values=field(j,0,"values"), iterations=field(j,0,"iterations"); double n=1;
        int valid=values>=0 && j->t[values].type==JARRAY && j->t[values].count>=1 && j->t[values].count<=64;
        for(int i=1;i<j->t[0].next;) { if(!text_eq(j,i,"id") && !text_eq(j,i,"op") && !text_eq(j,i,"values") && !text_eq(j,i,"iterations"))valid=0; i=j->t[i+1].next; }
        if(iterations>=0 && (!number(j,iterations,&n) || floor(n)!=n || n<1 || n>1000000000))valid=0;
        Job *job=calloc(1,sizeof(Job)); if(!job)exit(2); job->id=id; job->scale=scale; job->iterations=valid?(uint64_t)n:1;
        if(valid) for(int i=values+1;i<j->t[values].next;i=j->t[i].next) {
            if(!number(j,i,&value) || fabs(value)>1000000) { valid=0; break; } job->values[job->count++]=value;
        }
        if(!valid) { free(job); error(id,"invalid_request","Values and iterations must match the input schema."); return 0; }
        if(!loaded) { free(job); error(id,"not_loaded","Load the model first."); return 0; }
        pthread_mutex_lock(&state_lock); int busy=active_id!=0; pthread_mutex_unlock(&state_lock);
        if(busy) { free(job); error(id,"busy","Runner is busy."); return 0; }
        if(joinable) { pthread_join(thread,NULL); joinable=0; }
        pthread_mutex_lock(&state_lock); active_id=id; atomic_store(&cancelled,0); pthread_mutex_unlock(&state_lock);
        if(pthread_create(&thread,NULL,evaluate,job))exit(2); joinable=1;
    } else error(id,"unsupported_op","Unsupported operation.");
    return 0;
}
int main(void) {
    char hello[512]; snprintf(hello,sizeof(hello),"{\"event\":\"hello\",\"proto\":1,\"worker\":\"toy\",\"runner\":\"c\",\"pid\":%d,\"build\":\"template-1\",\"kinds\":[\"toy\"],\"ops\":[\"load\",\"ping\",\"status\",\"cancel\",\"unload\",\"evaluate\"]}",getpid()); send_json(hello);
    for(;;) {
        unsigned char h[4]; int got=exact(h,4,1);
        if(got==0) { finish_job(); return 0; } if(got<0)break;
        uint32_t n=(uint32_t)h[0]<<24|(uint32_t)h[1]<<16|(uint32_t)h[2]<<8|h[3];
        if(n==0 || n>MAX_FRAME)break;
        char *body=malloc((size_t)n+1); if(!body)break;
        if(exact(body,n,0)<0 || body[0]!=1) { free(body); break; } body[n]=0;
        JSON j; if(!json_parse(&j,body+1,(int)n-1)) { free(body); break; }
        int result=request(&j); free(body);
        if(result==1)return 0; if(result<0)break;
    }
    finish_job(); fputs("Invalid or truncated runner input.\n",stderr); return 2;
}
