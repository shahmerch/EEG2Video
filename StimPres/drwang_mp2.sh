****************************************
*************** ECE 5362 ***************
*********** Machine Problem 3 **********
*** Shahil Merchant/ Marcus Macesich ***
************** 11/15/2024 **************
****************************************


*** fetch cycle ******
 st=0 rt='[pc]-> mar'       imar rac=1 rn=3
 st=1 rt='[[mar]]-> mdr'    read
 st=2 rt='[mdr] -> ir'      omdr iir
 st=3 rt='[pc]+1 -> q'      rac=1 rn=3 ib p1 oadder
 st=4 rt='[q] -> pc'        oq wac=1 wn=3
    cond='ir1512' value=4 nst=70
    cond='ir1512' value=0 nst=100
    cond='ir158' value=0 nst=10 
    nst=0                          

*** halt cycle *****
 st=10 halt

*** single ops ***
 st=20
    cond='ir118' value=4 nst=21     
    nst=10    

*** NEG ********************************************              
 st=21
    cond='ir64' value=0 nst=22
    cond='ir64' value=1 nst=23  
    cond='ir64' value=2 nst=25     
    cond='ir64' value=3 nst=28  
    nst=31                       

***DST**************
*** direct ***
 st=22 rt='[Rn]->t4'                rac=3 it4
    nst=100

*** indirect ***
 st=23 rt='[Rn]->t5, [Rn]+1->q'     rac=3 it5 ib p1 oadder
 st=24 rt='[q]->Rn'                 oq wac=3
    nst=40

*** autoincrement ***
 st=25 rt='[Rn]->t1'                rac=3 it1
 st=26 rt='[t1]-1->q'               oa comp oadder
 st=27 rt='[q]->Rn, [q]->t5'        oq wac=3 it5
    nst=40

*** autodecrement ***
 st=28 rt='[Rn]->t1'                rac=3 it1
 st=29 rt='[t1]-1->q'               oa comp oadder
 st=30 rt='[q]->Rn, [q]->t5'        oq wac=3 it5
    nst=40


*** DST AM:4/5/6 *** read next word
 st=31 rt='[pc]-> mar'              imar rac=1 rn=3
 st=32 rt='[[mar]]-> mdr'           read
 st=33 rt='[mdr] -> t5'             omdr it5
 st=34 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=35 rt='[q] -> pc'               oq wac=1 wn=3
    cond='ir64' value=4 nst=37
    cond='ir64' value=5 nst=40
 st=36 rt='dst: Invalid AddrMod'    
    nst=10

*** DST AM:4 - Indexed ***
 st=37 rt='[t5] -> t1'              ot5 it1
 st=38 rt='[t1]+[Rn] -> q'          oa ib rac=2 oadder
 st=39 rt='[q] -> t5'               oq it5
    nst=40

*** Read Destination operand for AM:4/5 ***
 st=40 rt='[t5] -> mar'             ot5 imar
 st=41 rt='[[mar]] -> mdr'          read
 st=42 rt='[mdr] -> t4'             omdr it4
    nst=100


*** ********************************************
*** ********************************************
*** ********************************************
*** Double Operand: read source ***
 st=50                  rt='None'
    cond='ir108' value=0 nst=51
    cond='ir108' value=1 nst=52
    cond='ir108' value=2 nst=53
    cond='ir108' value=3 nst=55
    nst=38

*** SRC AM:0 - Register ***
 st=51 rt='[Rn]->t2'                rac=2 it2
    nst=100

*** SRC AM:1 - Register Indirect ***
 st=52 rt='[Rn]->t3'                rac=2 it3
    nst=70

*** SRC AM:2 - Autoincrement ***
 st=53 rt='[Rn]->t3, [Rn]+1->q'     rac=2 it3 ib p1 oadder
 st=54 rt='[q]->Rn'                 oq wac=2
    nst=70

*** SRC AM:3 - Autodecrement ***
 st=55 rt='[Rn]->t1'                rac=2 it1
 st=56 rt='[t1]-1->q'               oa comp oadder
 st=57 rt='[q]->Rn, [q]->t3'        oq wac=2 it3
    nst=70

*** SRC AM:4/5/6 *** read next word
 st=58 rt='[pc]-> mar'              imar rac=1 rn=3
 st=59 rt='[[mar]]-> mdr'           read
 st=60 rt='[mdr] -> t3'             omdr it3
 st=61 rt='[pc]+1 -> q'             rac=1 rn=3 ib p1 oadder
 st=62 rt='[q] -> pc'               oq wac=1 wn=3
    cond='ir108' value=4 nst=64
    cond='ir108' value=5 nst=70
    cond='ir108' value=6 nst=67
 st=63 rt='src: Invalid AddrMod'    
    nst=10

*** SRC AM:4 - Indexed ***
 st=64 rt='[t3] -> t1'              ot3 it1
 st=65 rt='[t1]+[Rn] -> q'          oa ib rac=2 oadder
 st=66 rt='[q] -> t3'               oq it3
    nst=70

*** SRC AM:6 - Absolute ***
**** EA is already in t3

*** SRC AM:6 - Immediate ***
 st=67 rt='[t3] -> t2'              ot3 it2
    nst=100

*** Read Source operand for AM:4/5 ***
 st=70 rt='[t3] -> mar'             ot3 imar
 st=71 rt='[[mar]] -> mdr'          read
 st=72 rt='[mdr] -> t2'             omdr it2
    nst=100

*** ********************************************
*** ********************************************
*** ********************************************

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
    cond='ir158' value=4 nst=152
    cond='ir158' value=5 nst=153
    cond='ir158' value=6 nst=154
    cond='ir158' value=7 nst=155
    cond='ir158' value=8 nst=165
    nst=10

*** src op ***
 st=70
    cond='ir118' value=0 nst=71     
    cond='ir118' value=1 nst=80   
    cond='ir118' value=2 nst=90    
    cond='ir118' value=3 nst=100   
    nst=10                          

*** direct ***
 st=71 rt='[Rs] -> t1'               it1 rac=2
    nst=110

*** indirect ***
 st=80 rt='[Rs] -> mar'              imar rac=2
 st=81 rt='[[mar]] -> mdr'           read 
 st=82 rt='[mdr] -> t1'              omdr  it1
    nst=110

*** autoincrement  ***
 st=90 rt='[Rs] -> mar'              imar rac=2
 st=91 rt='[[mar]] -> mdr'           read
 st=92 rt='[mdr] -> t1'              omdr it1
 st=93 rt='[Rs] + 1 -> q'            rac=2 ib p1 oadder
 st=94 rt='[q] -> Rs'                oq wac=2
    nst=110

*** autodecrement ***
 st=100 rt='[Rs] - 1 -> q'           rac=2 ib comp p1 oadder
 st=101 rt='[q] -> t5'               oq it5
 st=102 rt='0 - [t5] -> q'           ot5 ib comp oadder
 st=103 rt='[q] -> Rs'               oq wac=2
 st=104 rt='[Rs] -> mar'             imar rac=2
 st=105 rt='[[mar]] -> mdr'          read 
 st=106 rt='[mdr] -> t1'             omdr it1
    nst=110

*** dst op ***
 st=110
    cond='ir64' value=0 nst=111  
    cond='ir64' value=1 nst=120  
    cond='ir64' value=2 nst=130    
    cond='ir64' value=3 nst=140  
    nst=10                        
*** direct ***
 st=111 rt='[Rd] -> t2'              it2 rac=3
    nst=150

*** indirect ***
 st=120 rt='[Rd] -> mar'             imar rac=3
 st=121 rt='[[mar]] -> mdr'          read 
 st=122 rt='[mdr] -> t2'             ib omdr it2
    nst=150

*** autoincrement ***
 st=130 rt='[Rd] -> mar'             imar rac=3
 st=131 rt='[[mar]] -> mdr'          read 
 st=132 rt='[[mdr]] -> t2'           omdr it2
 st=133 rt='[Rd] + 1 -> q'           rac=3 ib p1 oadder
 st=134 rt='[q] -> Rd'               oq wac=3
    nst=150

*** autodecrement ***
 st=140 rt='[Rd] - 1 -> q'           rac=3 ib comp p1 oadder
 st=141 rt='[q] -> t5'               oq it5
 st=142 rt='0 - [t5] -> q'           ot5 ib comp oadder
 st=143 rt='[q] -> Rd'               oq wac=3
 st=144 rt='[Rd] -> mar'             imar rac=3
 st=145 rt='[[mar]] -> mdr'          read 
 st=146 rt='[[mdr]] -> t2'           omdr it2
    nst=150

*** swap ***
 st=150 rt='[t1] -> t3'              it3 ot1
 st=151 rt='[t2] -> t1'              it1 ot2
 st=152 rt='[t3] -> t2'              it2 ot3

*** write src ***
 st=153
    cond='ir118' value=0 nst=154  
    cond='ir118' value=1 nst=160  
    cond='ir118' value=2 nst=160    
    cond='ir118' value=3 nst=160   

*** direct ***
 st=154 rt='[t1] -> Rs'              ot1 wac=2
    nst=190

*** indirect ***
 st=160 rt='[Rs] -> mar'             imar rac=2
 st=161 rt='[t1] -> mdr'             ot1 imdr
 st=162 rt='[mdr] -> [mar]'          omdr write

    nst=190

*** autoincrement  ***
 st=170 rt='[Rs] -> mar'             imar rac=2
 st=171 rt='[t1] -> mdr'             ot1 imdr
 st=172 rt='[mdr] -> [mar]'          omdr write
    nst=190

*** autodecrement ***
 st=180 rt='[Rs] -> mar'             imar rac=2
 st=181 rt='[t1] -> mdr'             ot1 imdr
 st=182 rt='[mdr] -> [mar]'          omdr write
    nst=190

*** write dst ***
 st=190
    cond='ir64' value=0 nst=191     
    cond='ir64' value=1 nst=201     
    cond='ir64' value=2 nst=201    
    cond='ir64' value=3 nst=201     
    nst=10                          

*** direct ***
 st=191 rt='[t2] -> Rd'              ot2 wac=3
    nst=0

*** indirect ***
 st=201 rt='[t2] -> mdr'             ot2 imdr
 st=202 rt='[mdr] -> [mar]'          omdr write
 st=203 rt='[Rd] -> mar'             imar rac=3
    nst=0

*** autoincrement ***
 st=210 rt='[Rd] -> mar'             imar rac=3
 st=211 rt='[t2] -> mdr'             ot2 imdr
 st=212 rt='[mdr] -> [mar]'          omdr write
    nst=0

*** autodecrement ***
 st=220 rt='[Rd] -> mar'             imar rac=3
 st=221 rt='[t2] -> mdr'             ot2 imdr
 st=222 rt='[mdr] -> [mar]'          omdr write
    nst=0
