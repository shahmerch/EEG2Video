****************************************
*************** ECE 5362 ***************
*********** Machine Problem 3 **********
*** Shahil Merchant/ Marcus Macesich ***
************** 11/15/2024 **************
****************************************

*** Fetch ***
 st=0 rt='[pc]-> mar'       imar rac=1 rn=3
 st=1 rt='[[mar]]-> mdr'    read
 st=2 rt='[mdr] -> ir'      omdr iir
 st=3 rt='[pc]+1 -> q'      rac=1 rn=3 ib p1 oadder
 st=4 rt='[q] -> pc'        oq wac=1 wn=3
**special
    cond='ir158' value=0 nst=250
    cond='ir1512' value=0 nst=70
    nst=30

*** HALT ***
 st=10 halt

*** Double Operand - SRC Reg ********************************
 st=30                  rt='None'
    cond='ir108' value=0 nst=31
    cond='ir108' value=1 nst=32
    cond='ir108' value=2 nst=33
    cond='ir108' value=3 nst=35
    cond='ir108' value=4 nst=38
    cond='ir108' value=5 nst=38
    cond='ir108' value=6 nst=38
    nst=10

*** Source Mode 0: Register ***
 st=31 rt='[Rn]->t2'                rac=2 it2
    nst=70

*** Source Mode 1: Indirect ***
 st=32 rt='[Rn]->t3'                rac=2 it3
    nst=60

*** Source Mode 2: Autoincrement ***
 st=33 rt='[Rn]->t3, [Rn]+1->q'     rac=2 it3 ib p1 oadder
 st=34 rt='[q]->Rn'                 oq wac=2
    nst=60

*** Source Mode 3: Autodecrement ***
 st=35 rt='[Rn]->t1'                rac=2 it1
 st=36 rt='[t1]-1->q'               oa comp oadder
 st=37 rt='[q]->Rn, [q]->t3'        oq wac=2 it3
    nst=60

*** Source Mode 4,5,6 ***
 st=38 rt='[pc]-> mar'              imar rac=1 rn=3
 st=39 rt='[[mar]]-> mdr'           read
 st=40 rt='[mdr] -> t3'             omdr it3
 st=41 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=42 rt='[q] -> pc'               oq wac=1 wn=3
    cond='ir108' value=4 nst=44
    cond='ir108' value=5 nst=60
    cond='ir108' value=6 nst=47
 st=43 rt='Invalid Addressing of Source'    
    nst=10

*** Source Mode 4: Indexed ***
 st=44 rt='[t3] -> t1'              ot3 it1
 st=45 rt='[t1]+[Rn] -> q'          oa ib rac=2 oadder
 st=46 rt='[q] -> t3'               oq it3
    nst=60

*** Source Mode 5: Absolute ***

*** Source Mode 6: Immediate ***
 st=47 rt='[t3] -> t2'              ot3 it2
    nst=70

*** Sourec Op Mode 4&5 ***
 st=60 rt='[t3] -> mar'             ot3 imar
 st=61 rt='[[mar]] -> mdr'          read
 st=62 rt='[mdr] -> t2'             omdr it2
    nst=70

*** Double Operand - DST Reg / Single Op ********************************
 st=70                   rt='None'
    cond='ir64' value=0 nst=71
    cond='ir64' value=1 nst=72
    cond='ir64' value=2 nst=73
    cond='ir64' value=3 nst=75
    nst=78

*** Destination Mode 0: Register ***
 st=71 rt='[Rn]->t4'                rac=3 it4
    nst=100

*** Destination Mode 1: Indirect ***
 st=72 rt='[Rn]->t5'                rac=3 it5
    nst=90

*** Destination Mode 2: Autoincrement ***
 st=73 rt='[Rn]->t5, [Rn]+1->q'     rac=3 it5 ib p1 oadder
 st=74 rt='[q]->Rn'                 oq wac=3
    nst=90

*** Destination Mode 3: Autodecrement ***
 st=75 rt='[Rn]->t1'                rac=3 it1
 st=76 rt='[t1]-1->q'               oa comp oadder
 st=77 rt='[q]->Rn, [q]->t5'        oq wac=3 it5
    nst=90

*** Destination Mode 4,5,6 ***
 st=78 rt='[pc]-> mar'              imar rac=1 rn=3
 st=79 rt='[[mar]]-> mdr'           read
 st=80 rt='[mdr] -> t5'             omdr it5
 st=81 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=82 rt='[q] -> pc'               oq wac=1 wn=3
 
    cond='ir64' value=4 nst=84
    cond='ir64' value=5 nst=90
 st=83 rt='Invalid Addressing of Destination'    
    nst=10

*** Destination Mode 4: Indexed ***
 st=84 rt='[t5] -> t1'              ot5 it1
 st=85 rt='[t1]+[Rn] -> q'          oa ib rac=3 oadder
 st=86 rt='[q] -> t5'               oq it5
    nst=90

*** Destination ***
 st=90 rt='[t5] -> mar'             ot5 imar
 st=91 rt='[[mar]] -> mdr'          read
 st=92 rt='[mdr] -> t4'             omdr it4
    nst=100

*** Double Operations ********************************
 st=100                   rt='None'
** Add **
    cond='ir1512' value=1 nst=110
** Sub **
    cond='ir1512' value=2 nst=112
** Exg **
    cond='ir1512' value=4 nst=120
** Neg **
    cond='ir158' value=4 nst=180
** JSR **
    cond='ir158' value=7 nst=280
    nst=10

*** Add ***
 st=110 rt='[t4] -> t1'             ot4 it1
 st=111 rt='[t1]+[t2] -> q'         ot2 ib oa oadder newc newv 
    nst=160

*** Sub ***
 st=112 rt='[t4] -> t1'             ot4 it1
 st=113 rt='[t1]-[t2] -> q'         ot2 ib oa oadder comp p1
 st=114 rt='Carry'           
    cond='c' value=0 nst=115
    cond='c' value=1 nst=116
 st=115 rt='setc'                   setc
    nst=160
 st=116 rt='clrc'                   clrc
    nst=160

*** Exg ***
 st=120 rt='t2->q'                  ot2 ib oadder
    cond='ir108' value=0 nst=169

*** Mode 1,2,3,4,5 ***
 st=121 rt='[t3]->mar'             ot3 imar
 st=122 rt='[t4]->mdr'             ot4 imdr
 st=123 rt='[mdr]->[mar]'          write
    nst=170

*** Write to Destination ***
 st=160 rt='Write [q]->[t5]'
    cond='ir64' value=0 nst=164

*** Mode 1 to 5 Destination ***
 st=161 rt='[t5] -> mar'           ot5 imar
 st=162 rt='[q] -> mdr'            oq imdr newz newn
 st=163 rt='write'                 write
    nst=0

*** Mode 0 ***
 st=164 rt='q->Rn'                 oq wac=3 newz newn
    nst=0

*** Mode 0 ***
 st=169 rt='[t4]->Rn'              wac=2 ot4

*** all ***
 st=170 rt='Write [q]->[t5]'
    cond='ir64' value=0 nst=174

*** Mode 1 to 5 ***
 st=171 rt='[t5] -> mar'           ot5 imar
 st=172 rt='[q] -> mdr'            oq imdr 
 st=173 rt='write'                 write
    nst=0

*** Mode 0 ***
 st=174 rt='q->Rn'                 oq wac=3 
    nst=0

*** Single Operand ******************************************
*** NEG ***         
 st=180
    cond='ir64' value=0 nst=181     
    nst=190                      

*** Direct ***
 st=181 rt='[Rk] -> t2'               it1 rac=3
 st=182 rt='0 - [t1] -> q'            ib comp p1 oadder newc newv
 st=183 rt='[q] -> Rk'                oq wac=3 newz newn
    nst=0

*** Indirect ***
 st=190 rt='[Rk] -> mar'              imar ot5
 st=191 rt='[[mar]] -> mdr'           read
 st=192 rt='[mdr] -> t1'              omdr it1
 st=193 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=194 rt='[q] -> mdr'               oq imdr newz newn
 st=195 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** Special Instructions *************************************
 st=250                  rt='None'   
**Branch**
    cond='ibrch' value=1 nst=260
**RTS**
    cond='ir75' value=4 nst=290
    nst=10


*** Branch ***
 st=260 rt='Check ibrch'
    cond='ibrch' value=0 nst=142

 st=261 rt='ibrch=1 XOR IR5'
    cond='ir5' value=0 nst=143
    cond='ir5' value=1 nst=150

 st=262 rt='ibrch=0 XOR IR5'
    cond='ir5' value=0 nst=150
    cond='ir5' value=1 nst=143

 st=263 rt='[PC]->mar'             rac=1 rn=3 it1 imar
 st=264 rt='[PC]->t1'              rac=1 rn=3 it1
 st=265 rt='[[mar]] -> mdr'        read
 st=266 rt='[t1]+[mdr]+1->q'       omdr oa ib p1 oadder
 st=267 rt='[q]->pc'               oq wac=1 wn=3
    nst=0

 st=270 rt='PC+1->q'               rac=1 rn=3 ib p1 oadder
 st=271 rt='q->pc'                 oq wac=1 wn=3
    nst=0

*** JSR ***
 st=280 rt='[SP] - 1 -> q'            rac=1 rn=2 ib comp p1 oadder
 st=281 rt='[q] -> t5'                oq it2
 st=282 rt='0 -[t5] -> q'             ot2 ib comp oadder
 st=283 rt='[q] -> Rk'                oq wac=1 wn = 2
 st=284 rt='[pc] -> mdr'              rac=1 rn=3 imdr
 st=285 rt='[SP] -> mar'              rac=1 rn=2 imar
 st=286 rt='[mdr] -> [mar]'           omdr write
 st=287 rt='[t5] -> pc'               ot5 wac=1 wn=3
   nst=0

*** RTS ***
 st=290 rt='[SP] -> mar'              rac=1 rn=2 imar
 st=291 rt='[[mar]] -> mdr'           read 
 st=292 rt='[mdr] -> pc'              omdr wac=1 wn=3
 st=293 rt='[SP] + 1 -> q'            rac=1 rn=2 ib p1 oadder wac=1 wn=2
 st=294 rt='[q] -> SP'                oq wac=1 wn=2
   nst=0