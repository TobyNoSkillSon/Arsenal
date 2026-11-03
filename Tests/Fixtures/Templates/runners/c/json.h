/* SPDX-License-Identifier: MIT
 * Small recursive JSON tokenizer for this CPU example. Not a model dependency.
 * Tokens preserve spans and child boundaries; keys and values remain distinct.
 */
#ifndef TOY_JSON_H
#define TOY_JSON_H
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int type, start, end, next, count; } Token;
typedef struct { const char *s; int pos, len, used; Token t[2048]; } JSON;
enum { JOBJECT=1, JARRAY, JSTRING, JNUMBER, JLITERAL };
static void spaces(JSON *j) { while (j->pos < j->len && strchr(" \n\r\t", j->s[j->pos])) j->pos++; }
static int parse_value(JSON *j, int depth) {
    spaces(j);
    if (depth > 32 || j->used >= 2048 || j->pos >= j->len) return -1;
    int index = j->used++;
    Token *t = &j->t[index]; t->start = j->pos;
    char c = j->s[j->pos++];
    if (c == '{' || c == '[') {
        t->type = c == '{' ? JOBJECT : JARRAY;
        char close = c == '{' ? '}' : ']'; spaces(j);
        if (j->pos < j->len && j->s[j->pos] == close) { j->pos++; goto done; }
        for (;;) {
            if (c == '{') {
                int key = parse_value(j, depth+1);
                if (key < 0 || j->t[key].type != JSTRING) return -1;
                spaces(j); if (j->pos >= j->len || j->s[j->pos++] != ':') return -1;
            }
            if (parse_value(j, depth+1) < 0) return -1;
            t->count++; spaces(j);
            if (j->pos >= j->len) return -1;
            char separator = j->s[j->pos++];
            if (separator == close) break;
            if (separator != ',') return -1;
        }
    } else if (c == '"') {
        t->type = JSTRING;
        for (;;) {
            if (j->pos >= j->len) return -1;
            unsigned char ch = (unsigned char)j->s[j->pos++];
            if (ch == '"') break;
            if (ch < 32) return -1;
            if (ch == '\\') {
                if (j->pos >= j->len) return -1;
                ch = (unsigned char)j->s[j->pos++];
                if (ch == 'u') {
                    for (int n=0;n<4;n++) if (j->pos >= j->len || !isxdigit((unsigned char)j->s[j->pos++])) return -1;
                } else if (!strchr("\"\\/bfnrt", ch)) return -1;
            }
        }
    } else {
        while (j->pos < j->len && !strchr(" ,]}\r\n\t", j->s[j->pos])) j->pos++;
        int n = j->pos - t->start;
        if ((n==4 && !memcmp(j->s+t->start,"true",4)) || (n==5 && !memcmp(j->s+t->start,"false",5)) || (n==4 && !memcmp(j->s+t->start,"null",4))) t->type = JLITERAL;
        else {
            // Check JSON's number grammar before strtod (no NaN, hex or '+1').
            int p=t->start, end=j->pos;
            if (p<end && j->s[p]=='-') p++;
            if (p>=end) return -1;
            if (j->s[p]=='0') p++;
            else { if (j->s[p]<'1' || j->s[p]>'9') return -1; while(p<end && isdigit((unsigned char)j->s[p])) p++; }
            if(p<end && j->s[p]=='.') { p++; int first=p; while(p<end && isdigit((unsigned char)j->s[p])) p++; if(p==first)return -1; }
            if(p<end && (j->s[p]=='e'||j->s[p]=='E')) { p++; if(p<end && (j->s[p]=='+'||j->s[p]=='-')) p++; int first=p; while(p<end && isdigit((unsigned char)j->s[p]))p++; if(p==first)return -1; }
            if(p!=end)return -1;
            t->type=JNUMBER;
        }
    }
done:
    t->end=j->pos; t->next=j->used; return index;
}
static int utf8(const char *s, int len) {
    for (int i=0; i<len;) {
        unsigned char first=(unsigned char)s[i++];
        if (first<128) continue;
        unsigned int cp; int tail;
        if (first>=0xc2 && first<=0xdf) { cp=first&31; tail=1; }
        else if (first>=0xe0 && first<=0xef) { cp=first&15; tail=2; }
        else if (first>=0xf0 && first<=0xf4) { cp=first&7; tail=3; }
        else return 0;
        for (int n=0; n<tail; n++) {
            if (i>=len) return 0;
            unsigned char b=(unsigned char)s[i++];
            if ((b&0xc0)!=0x80) return 0;
            cp=(cp<<6)|(b&63);
        }
        if ((tail==1 && cp<128) || (tail==2 && cp<2048) || (tail==3 && cp<65536) || cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff)) return 0;
    }
    return 1;
}
static int json_parse(JSON *j, const char *s, int len) {
    if (!utf8(s,len)) return 0;
    memset(j,0,sizeof(*j)); j->s=s; j->len=len;
    if(parse_value(j,0)!=0)return 0;
    spaces(j); return j->pos==len && j->t[0].type==JOBJECT;
}
static int text_eq(JSON *j, int i, const char *s) {
    return i>=0 && j->t[i].type==JSTRING && j->t[i].end-j->t[i].start-2==(int)strlen(s) && !memcmp(j->s+j->t[i].start+1,s,strlen(s));
}
static int field(JSON *j,int object,const char *name) {
    if(object<0 || j->t[object].type!=JOBJECT)return -1;
    for(int i=object+1;i<j->t[object].next;) {
        int value=i+1; if(text_eq(j,i,name))return value; i=j->t[value].next;
    }
    return -1;
}
static int number(JSON *j,int i,double *out) {
    if(i<0 || j->t[i].type!=JNUMBER)return 0;
    *out=strtod(j->s+j->t[i].start,NULL); return isfinite(*out);
}
#endif
