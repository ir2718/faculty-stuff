.file	"zad6.cpp"
	.intel_syntax noprefix
	.text
	.section	.text$Base::Base(),"x"
	.linkonce discard
	.align 2
	.globl	Base::Base()
	.def	Base::Base();	.scl	2;	.type	32;	.endef
	.seh_proc	Base::Base()
Base::Base():
.LFB29:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 32
	.seh_stackalloc	32
	.seh_endprologue
	mov	QWORD PTR 16[rbp], rcx
	lea	rdx, vtable for Base[rip 16]	// postavljanje pokazivaca na virtualnu tablicu
	mov	rax, QWORD PTR 16[rbp]
	mov	QWORD PTR [rax], rdx
	mov	rcx, QWORD PTR 16[rbp]
	call	Base::metoda()			// poziv metode
	nop
	add	rsp, 32
	pop	rbp
	ret
	.seh_endproc
	.section .rdata,"dr"
.LC0:
	.ascii "ja sam bazna implementacija!\0"
	.section	.text$Base::virtualnaMetoda(),"x"
	.linkonce discard
	.align 2
	.globl	Base::virtualnaMetoda()
	.def	Base::virtualnaMetoda();	.scl	2;	.type	32;	.endef
	.seh_proc	Base::virtualnaMetoda()
Base::virtualnaMetoda():
.LFB31:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 32
	.seh_stackalloc	32
	.seh_endprologue
	mov	QWORD PTR 16[rbp], rcx
	lea	rcx, .LC0[rip]
	call	puts
	nop
	add	rsp, 32
	pop	rbp
	ret
	.seh_endproc
	.section .rdata,"dr"
.LC1:
	.ascii "Metoda kaze: \0"
	.section	.text$Base::metoda(),"x"
	.linkonce discard
	.align 2
	.globl	Base::metoda()
	.def	Base::metoda();	.scl	2;	.type	32;	.endef
	.seh_proc	Base::metoda()
Base::metoda():
.LFB32:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 32
	.seh_stackalloc	32
	.seh_endprologue
	mov	QWORD PTR 16[rbp], rcx
	lea	rcx, .LC1[rip]
	call	printf
	mov	rax, QWORD PTR 16[rbp]
	mov	rax, QWORD PTR [rax]
	mov	rax, QWORD PTR [rax]
	mov	rcx, QWORD PTR 16[rbp]
	call	rax
	nop
	add	rsp, 32
	pop	rbp
	ret
	.seh_endproc
	.section	.text$Derived::Derived(),"x"
	.linkonce discard
	.align 2
	.globl	Derived::Derived()
	.def	Derived::Derived();	.scl	2;	.type	32;	.endef
	.seh_proc	Derived::Derived()
Derived::Derived():
.LFB35:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 32
	.seh_stackalloc	32
	.seh_endprologue
	mov	QWORD PTR 16[rbp], rcx
	mov	rax, QWORD PTR 16[rbp]
	mov	rcx, rax
	call	Base::Base()				// poziv konstruktora nadklase
	lea	rdx, vtable for Derived[rip 16]		// postavljanje pokazivaca na virtualnu tablicu
	mov	rax, QWORD PTR 16[rbp]
	mov	QWORD PTR [rax], rdx
	mov	rax, QWORD PTR 16[rbp]
	mov	rcx, rax
	call	Base::metoda()				// poziv metode
	nop
	add	rsp, 32
	pop	rbp
	ret
	.seh_endproc
	.section .rdata,"dr"
	.align 8
.LC2:
	.ascii "ja sam izvedena implementacija!\0"
	.section	.text$Derived::virtualnaMetoda(),"x"
	.linkonce discard
	.align 2
	.globl	Derived::virtualnaMetoda()
	.def	Derived::virtualnaMetoda();	.scl	2;	.type	32;	.endef
	.seh_proc	Derived::virtualnaMetoda()
Derived::virtualnaMetoda():
.LFB36:
	push	rbp
	.seh_pushreg	rbp
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 32
	.seh_stackalloc	32
	.seh_endprologue
	mov	QWORD PTR 16[rbp], rcx
	lea	rcx, .LC2[rip]
	call	puts
	nop
	add	rsp, 32
	pop	rbp
	ret
	.seh_endproc
	.def	__main;	.scl	2;	.type	32;	.endef
	.text
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
	.seh_proc	main
main:
.LFB37:
	push	rbp
	.seh_pushreg	rbp
	push	rsi
	.seh_pushreg	rsi
	push	rbx
	.seh_pushreg	rbx
	mov	rbp, rsp
	.seh_setframe	rbp, 0
	sub	rsp, 48
	.seh_stackalloc	48
	.seh_endprologue
	call	__main
	mov	ecx, 8
.LEHB0:
	call	operator new(unsigned long long)
.LEHE0:
	mov	rbx, rax
	mov	rcx, rbx
.LEHB1:
	call	Derived::Derived()	// poziv derived konstruktora
.LEHE1:
	mov	QWORD PTR -8[rbp], rbx
	mov	rax, QWORD PTR -8[rbp]
	mov	rcx, rax
.LEHB2:
	call	Base::metoda()		// ponovni poziv metode
	mov	eax, 0
	jmp	.L10
.L9:
	mov	rsi, rax
	mov	edx, 8
	mov	rcx, rbx
	call	operator delete(void*, unsigned long long)
	mov	rax, rsi
	mov	rcx, rax
	call	_Unwind_Resume
.LEHE2:
.L10:
	add	rsp, 48
	pop	rbx
	pop	rsi
	pop	rbp
	ret
	.def	__gxx_personality_seh0;	.scl	2;	.type	32;	.endef
	.seh_handler	__gxx_personality_seh0, @unwind, @except
	.seh_handlerdata
.LLSDA37:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE37-.LLSDACSB37
.LLSDACSB37:
	.uleb128 .LEHB0-.LFB37
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB37
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L9-.LFB37
	.uleb128 0
	.uleb128 .LEHB2-.LFB37
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSE37:
	.text
	.seh_endproc
	.globl	vtable for Derived
	.section	.rdata$vtable for Derived,"dr"
	.linkonce same_size
	.align 8
vtable for Derived:
	.quad	0
	.quad	typeinfo for Derived
	.quad	Derived::virtualnaMetoda()
	.globl	vtable for Base
	.section	.rdata$vtable for Base,"dr"
	.linkonce same_size
	.align 8
vtable for Base:
	.quad	0
	.quad	typeinfo for Base
	.quad	Base::virtualnaMetoda()
	.globl	typeinfo for Derived
	.section	.rdata$typeinfo for Derived,"dr"
	.linkonce same_size
	.align 8
typeinfo for Derived:
	.quad	vtable for __cxxabiv1::__si_class_type_info 16
	.quad	typeinfo name for Derived
	.quad	typeinfo for Base
	.globl	typeinfo name for Derived
	.section	.rdata$typeinfo name for Derived,"dr"
	.linkonce same_size
	.align 8
typeinfo name for Derived:
	.ascii "7Derived\0"
	.globl	typeinfo for Base
	.section	.rdata$typeinfo for Base,"dr"
	.linkonce same_size
	.align 8
typeinfo for Base:
	.quad	vtable for __cxxabiv1::__class_type_info 16
	.quad	typeinfo name for Base
	.globl	typeinfo name for Base
	.section	.rdata$typeinfo name for Base,"dr"
	.linkonce same_size
typeinfo name for Base:
	.ascii "4Base\0"
	.ident	"GCC: (x86_64-win32-seh-rev0, Built by MinGW-W64 project) 8.1.0"
	.def	puts;	.scl	2;	.type	32;	.endef
	.def	printf;	.scl	2;	.type	32;	.endef
	.def	operator new(unsigned long long);	.scl	2;	.type	32;	.endef
	.def	operator delete(void*, unsigned long long);	.scl	2;	.type	32;	.endef
	.def	_Unwind_Resume;	.scl	2;	.type	32;	.endef

----------------------------------------------
Prvo se poziva konstruktor klase Derived koja odmah poziva konstruktor klase Base. Zatim se u konstruktoru klase Base postavlja pokazivac na virtualnu tablicu klase Base pa se poziva metoda i virtualnaMetoda što objašnjava prvi ispis. Zatim se tok programa vraca u konstruktor Base, konstruktor završava i vraca se natrag u konstruktor klase Derived. Ponovo se postavlja pokazivac na virtualnu tablicu, ali ovaj put na virtualnu tablicu konstruktora klase Derived. Zatim se poziva metoda i virtualnaMetoda koja ovaj puta ima drukciji ispis jer se u virtualnoj tablici nalazi funkcija virtualnaMetoda klase Derived. Na kraju se još jednom poziva metoda koja poziva virtualnuMetodu klase Derived jer se pokazivac na virtualnu tablicu nije mijenjao.