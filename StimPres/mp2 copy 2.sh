****************************************
*************** ECE 5362 ***************
*********** Machine Problem 3 **********
*** Shahil Merchant/ Marcus Macesich ***
************** 11/15/2024 **************
****************************************

*** Start fetch cycle ***
 st=0 rt='[pc]-> mar'       imar rac=1 rn=3
 st=1 rt='[[mar]]-> mdr'    read
 st=2 rt='[mdr] -> ir'      omdr iir
 st=3 rt='[pc]+1 -> q'      rac=1 rn=3 ib p1 oadder
 st=4 rt='[q] -> pc'        oq wac=1 wn=3
    cond='ir158' value=0 nst=20
    cond='ir1512' value=0 nst=70
    nst=30

*** HALT execution cycle ***
 st=10 halt

*** Branch / Special Instructions ***
 st=20                  rt='None'
    cond='ibrch' value=1 nst=170
    cond='ir75' value=4 nst=300
    nst=10

*** Double Operand: read source ***
 st=30                  rt='None'
    cond='ir108' value=0 nst=31
    cond='ir108' value=1 nst=32
    cond='ir108' value=2 nst=33
    cond='ir108' value=3 nst=35
    nst=38

*** SRC AM:0 - Register ***
 st=31 rt='[Rn]->t2'                rac=2 it2
    nst=70

*** SRC AM:1 - Register Indirect ***
 st=32 rt='[Rn]->t3'                rac=2 it3
    nst=60

*** SRC AM:2 - Autoincrement ***
 st=33 rt='[Rn]->t3, [Rn]+1->q'     rac=2 it3 ib p1 oadder
 st=34 rt='[q]->Rn'                 oq wac=2
    nst=60

*** SRC AM:3 - Autodecrement ***
 st=35 rt='[Rn]->t1'                rac=2 it1
 st=36 rt='[t1]-1->q'               oa comp oadder
 st=37 rt='[q]->Rn, [q]->t3'        oq wac=2 it3
    nst=60

*** SRC AM:4/5/6 *** read next word
 st=38 rt='[pc]-> mar'              imar rac=1 rn=3
 st=39 rt='[[mar]]-> mdr'           read
 st=40 rt='[mdr] -> t3'             omdr it3
 st=41 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=42 rt='[q] -> pc'               oq wac=1 wn=3
    cond='ir108' value=4 nst=44
    cond='ir108' value=5 nst=60
    cond='ir108' value=6 nst=47
 st=43 rt='src: Invalid AddrMod'    
    nst=10

*** SRC AM:4 - Indexed ***
 st=44 rt='[t3] -> t1'              ot3 it1
 st=45 rt='[t1]+[Rn] -> q'          oa ib rac=2 oadder
 st=46 rt='[q] -> t3'               oq it3
    nst=60

*** SRC AM:6 - Absolute ***
**** EA is already in t3

*** SRC AM:6 - Immediate ***
 st=47 rt='[t3] -> t2'              ot3 it2
    nst=70

*** Read Source operand for AM:4/5 ***
 st=60 rt='[t3] -> mar'             ot3 imar
 st=61 rt='[[mar]] -> mdr'          read
 st=62 rt='[mdr] -> t2'             omdr it2
    nst=70

*** Single/ Double Operand: read destination ***
 st=70                   rt='None'
    cond='ir64' value=0 nst=71
    cond='ir64' value=1 nst=72
    cond='ir64' value=2 nst=73
    cond='ir64' value=3 nst=75
    nst=78

*** DST AM:0 - Register ***
 st=71 rt='[Rn]->t4'                rac=3 it4
    nst=100

*** DST AM:1 - Register Indirect ***
 st=72 rt='[Rn]->t5'                rac=3 it5
    nst=90

*** DST AM:2 - Autoincrement ***
 st=73 rt='[Rn]->t5, [Rn]+1->q'     rac=3 it5 ib p1 oadder
 st=74 rt='[q]->Rn'                 oq wac=3
    nst=90

*** DST AM:3 - Autodecrement ***
 st=75 rt='[Rn]->t1'                rac=3 it1
 st=76 rt='[t1]-1->q'               oa comp oadder
 st=77 rt='[q]->Rn, [q]->t5'        oq wac=3 it5
    nst=90

*** DST AM:4/5/6 *** read next word
 st=78 rt='[pc]-> mar'              imar rac=1 rn=3
 st=79 rt='[[mar]]-> mdr'           read
 st=80 rt='[mdr] -> t5'             omdr it5
 st=81 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=82 rt='[q] -> pc'               oq wac=1 wn=3
 
    cond='ir64' value=4 nst=84
    cond='ir64' value=5 nst=90
 st=83 rt='dst: Invalid AddrMod'    
    nst=10

*** DST AM:4 - Indexed ***
 st=84 rt='[t5] -> t1'              ot5 it1
 st=85 rt='[t1]+[Rn] -> q'          oa ib rac=3 oadder
 st=86 rt='[q] -> t5'               oq it5
    nst=90

*** DST AM:6 - Absolute ***
**** EA is already in t5

*** Read Destination operand for AM:4/5 ***
 st=90 rt='[t5] -> mar'             ot5 imar
 st=91 rt='[[mar]] -> mdr'          read
 st=92 rt='[mdr] -> t4'             omdr it4
    nst=100

*** Test OpCode ***
 st=100                   rt='None'
    cond='ir1512' value=1 nst=110
    cond='ir1512' value=2 nst=115
    cond='ir1512' value=3 nst=120
    cond='ir1512' value=4 nst=130
    cond='ir1512' value=5 nst=121
    cond='ir1512' value=6 nst=122
    cond='ir158' value=1 nst=149
    cond='ir158' value=2 nst=150
    cond='ir158' value=3 nst=151
    cond='ir158' value=4 nst=221
    cond='ir158' value=5 nst=153
    cond='ir158' value=6 nst=154
    cond='ir158' value=7 nst=290
    cond='ir158' value=8 nst=165
    nst=10

*** Op: Add ***
 st=110 rt='[t4] -> t1'             ot4 it1
 st=111 rt='[t1]+[t2] -> q'         ot2 ib oa oadder newc newv 
    nst=190

*** Op: Sub ***
 st=115 rt='[t4] -> t1'             ot4 it1
 st=116 rt='[t1]-[t2] -> q'         ot2 ib oa oadder comp p1
 st=117 rt='barrow check'           
    cond='c' value=0 nst=118
    cond='c' value=1 nst=119
 st=118 rt='setc'                   setc
    nst=190
 st=119 rt='clrc'                   clrc
    nst=190

*** Op: Move ***
 st=120 rt='[t2] -> q'              ot2 iq clrc clrv
    nst=190

*** Op: OR ***
 st=121 rt='t2 OR t4 -> q'          ot2 ot4 ib oadder clrc clrv
    nst=190

*** Op: AND ***
 st=122 rt='ones compl of t2 -> q'  ot2 ib comp oadder
 st=123 rt='[q]+t4'                 ot4 ib comp oadder
 st=124 rt='[q]->t4'                oq it2
    nst=190

*** Op: Exg ***
 st=130 rt='t2->q'                  ot2 ib oadder

*** Write dst to src ***
 st=131                  rt='None'
    cond='ir108' value=0 nst=133
    nst=134

*Register: Mode 0*
 st=133 rt='[t4]->Rn'              wac=2 ot4
    nst=200

*Indirect: Mode 1,2,3,4,5*
 st=134 rt='[t3]->mar'             ot3 imar
 st=135 rt='[t4]->mdr'             ot4 imdr
 st=136 rt='[mdr]->[mar]'          write
    nst=200


    
******* Branch instructions ***************
 st=170 rt='test ibrch'
    cond='ibrch' value=0 nst=172

 st=171 rt='ibrch=1 XOR with IR5'
    cond='ir5' value=0 nst=173
    cond='ir5' value=1 nst=180

 st=172 rt='ibrch=0 XOR IR5'
    cond='ir5' value=0 nst=180
    cond='ir5' value=1 nst=173

 st=173 rt='PC->mar, PC->t1'       rac=1 rn=3 it1 imar
 st=174 rt='read'                  read
 st=175 rt='t1+1+mdr->q'           omdr oa ib p1 oadder
 st=176 rt='q->pc'                 oq wac=1 wn=3
    nst=0

 st=180 rt='PC+1->q'               rac=1 rn=3 ib p1 oadder
 st=181 rt='q->pc'                 oq wac=1 wn=3
    nst=0

********** Write result to DST ***************
 st=190 rt='WriteBack [q]->[t5]'
    cond='ir64' value=0 nst=195

**** --- Addr Mode 1-5: 6 is invalid for destination
 st=191 rt='[t5] -> mar'           ot5 imar
 st=192 rt='[q] -> mdr'            oq imdr newz newn
 st=193 rt='write'                 write
    nst=0

*** --- Addr Mode 0: Register
 st=195 rt='q->Rn'                 oq wac=3 newz newn
    nst=0


 st=200 rt='WriteBack [q]->[t5]'
    cond='ir64' value=0 nst=195

**** --- Addr Mode 1-5: 6 is invalid for destination
 st=201 rt='[t5] -> mar'           ot5 imar
 st=202 rt='[q] -> mdr'            oq imdr 
 st=203 rt='write'                 write
    nst=0

*** --- Addr Mode 0: Register
 st=205 rt='q->Rn'                 oq wac=3 
    nst=0



****** Add single-operand instructions here ******

*** NEG ********************************************              
 st=221
    cond='ir64' value=0 nst=230     
    nst=240                      

*** direct ***
 st=230 rt='[Rk] -> t2'               it1 rac=3
 st=231 rt='0 - [t1] -> q'            ib comp p1 oadder newc newv
 st=232 rt='[q] -> Rk'                oq wac=3 newz newn
    nst=0

*** indirect ***
 st=240 rt='[Rk] -> mar'              imar ot5
 st=241 rt='[[mar]] -> mdr'           read
 st=242 rt='[mdr] -> t1'              omdr it1
 st=243 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=244 rt='[q] -> mdr'               oq imdr newz newn
 st=245 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** autoincrement ***
 st=250 rt='[Rk] -> mar'              imar rac=3
 st=251 rt='[[mar]] -> mdr'           read
 st=252 rt='[Rk] + 1 -> q'            rac=3 ib p1 oadder
 st=253 rt='[q] -> Rk'                oq wac=3
 st=254 rt='[mdr] -> t1'              omdr it1
 st=255 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=256 rt='[q] -> mdr'               oq imdr newz newn
 st=257 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** autodecrement ***
 st=260 rt='[Rk] - 1 -> q'            rac=3 ib comp p1 oadder
 st=261 rt='[q] -> t5'                oq it5
 st=262 rt='0 -[t5] -> q'             ot5 ib comp oadder
 st=263 rt='[q] -> Rk'                oq wac=3
 st=264 rt='[Rk] -> mar'              imar rac=3
 st=265 rt='[[mar]] -> mdr'           read
 st=266 rt='[mdr] -> t1'              omdr it1
 st=267 rt='0 - [t1] -> q'            ot1 ib comp p1 oadder newc newv
 st=268 rt='[q] -> mdr'               oq imdr newz newn
 st=269 rt='[mdr] -> [mar]'           omdr write
    nst=0

*** JSR Instruction ***
 st=290 rt='[SP] - 1 -> q'            rac=1 rn=2 ib comp p1 oadder
 st=291 rt='[q] -> t5'                oq it2
 st=292 rt='0 -[t5] -> q'             ot2 ib comp oadder
 st=293 rt='[q] -> Rk'                oq wac=1 wn = 2
 st=294 rt='[pc] -> mdr'              rac=1 rn=3 imdr
 st=295 rt='[SP] -> mar'              rac=1 rn=2 imar
 st=296 rt='[mdr] -> [mar]'           omdr write
 st=297 rt='[t5] -> pc'               ot5 wac=1 wn=3
   nst=0

*** RTS Instruction ***
 st=300 rt='[SP] -> mar'              rac=1 rn=2 imar
 st=301 rt='[[mar]] -> mdr'           read 
 st=302 rt='[mdr] -> pc'              omdr wac=1 wn=3
 st=303 rt='[SP] + 1 -> q'            rac=1 rn=2 ib p1 oadder wac=1 wn=2
 st=304 rt='[q] -> SP'                oq wac=1 wn=2
   nst=0
