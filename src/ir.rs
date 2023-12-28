use crate::sexpr::*;
use anyhow::{anyhow, Result};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
enum IrErrors {
    #[error("Duplicate variable name")]
    DuplicateName,

    #[error("Unknown expression")]
    UnknownExpression,

    #[error("Bad binop types")]
    BadBinopTypes,

    #[error("Forbidden variable declaration")]
    ForbiddenVariableDeclaration,

    #[error("Bad variable init type")]
    BadVariableInitType,

    #[error("Bad variable set type")]
    BadVariableSetType,

    #[error("Variable not found")]
    VariableNotFound,

    #[error("Expected boolean condition")]
    ExpectedBooleanCondition,

    #[error("Break used outside of loop")]
    BreakOutsideLoop,

    #[error("Continue used outside of loop")]
    ContinueOutsideLoop,

    #[error("Bad condition type")]
    BadConditionType,

    #[error("Bad body type")]
    BadBodyType,

    #[error("Missing return type")]
    MissingReturnType,

    #[error("Function not found")]
    FunctionNotFound,

    #[error("bad unop types")]
    BadUnopTypes,

    #[error("Trying to reassing const variable.")]
    AssignToConst,

    #[error("Invalid function arguments")]
    InvalidFunctionArguments,
}

#[derive(Copy, Clone, PartialEq, Debug, Hash, Eq)]
pub enum Type {
    Int,
    Byte,
    BytePtr,
    Void,
}

impl std::str::FromStr for Type {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "void" => Ok(Self::Void),
            "byte" => Ok(Self::Byte),
            "int" => Ok(Self::Int),
            _ => Err(anyhow!("unrecognized type name")),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Hash, Eq)]
pub enum Binop {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    And,
    Or,
    GT,
    LE,
    EQ,
    NE,
    LT,
    GE,
}

fn calc_binop(op: Binop, a: i64, b: i64) -> Result<i64> {
    match op {
        Binop::Plus => Ok(a + b),
        Binop::Minus => Ok(a - b),
        Binop::Multiply => Ok(a * b),
        Binop::Divide => {
            if b != 0 {
                Ok(a / b)
            } else {
                // Handle division by zero error, for now, returning 0
                Ok(0)
            }
        }
        _ => Err(anyhow!("cannot do binop")),
    }
}

impl std::str::FromStr for Binop {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "+" => Ok(Self::Plus),
            "-" => Ok(Self::Minus),
            "/" => Ok(Self::Divide),
            "*" => Ok(Self::Multiply),
            "%" => Ok(Self::Modulo),
            "gt" => Ok(Self::GT),
            "le" => Ok(Self::LE),
            "lt" => Ok(Self::LT),
            "eq" => Ok(Self::EQ),
            "ne" => Ok(Self::NE),
            "ge" => Ok(Self::GE),
            "and" => Ok(Self::And),
            "or" => Ok(Self::Or),
            _ => Err(anyhow!("unsupported binary operation")),
        }
    }
}

impl std::fmt::Display for Binop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minus => write!(f, "-"),
            Self::Multiply => write!(f, "*"),
            Self::Divide => write!(f, "/"),
            Self::Plus => write!(f, "+"),
            Self::GT => write!(f, "gt"),
            Self::LE => write!(f, "le"),
            Self::EQ => write!(f, "eq"),
            Self::LT => write!(f, "lt"),
            Self::Modulo => write!(f, "%"),
            Self::NE => write!(f, "ne"),
            Self::GE => write!(f, "ge"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Unop {
    Minus,
    Not,
}

impl std::str::FromStr for Unop {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "-" => Ok(Self::Minus),
            "not" => Ok(Self::Not),
            _ => Err(anyhow!("unsupported binary operation")),
        }
    }
}

impl std::fmt::Display for Unop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minus => write!(f, "-"),
            Self::Not => write!(f, "not"),
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum Instruction {
    Constant(i64, i64),
    Binop(Binop, i64, i64, i64, bool),
    Mov(i64, i64),
    Jmpf(i64, i64),
    Jmp(i64),
    Ret(i64),
    Call(i64, i64, i64, i64),
    Unop(Unop, i64, i64),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(a, b) => write!(f, "const {} {}", a, b),
            Self::Binop(ty, lhs, rhs, dst, is_byte) => {
                if *is_byte {
                    write!(f, "binop8 {} {} {} {}", ty, lhs, rhs, dst)
                } else {
                    write!(f, "binop {} {} {} {}", ty, lhs, rhs, dst)
                }
            }
            Self::Mov(a, b) => write!(f, "mov {} {}", a, b),
            Self::Jmp(l) => write!(f, "jmp L{}", l),
            Self::Jmpf(a, b) => write!(f, "jmpf {} L{}", a, b),
            Self::Ret(a) => write!(f, "ret {}", a),
            Self::Call(a, b, c, d) => write!(f, "call {} {} {} {}", a, b, c, d),
            Self::Unop(a, b, c) => write!(f, "unop {} {} {}", a, b, c),
        }
    }
}

type Types = Vec<Type>;
type TypeIndex = (Types, i64);

#[derive(Debug, Clone, PartialEq)]
enum VariableType {
    Var,
    Const,
    Function,
}

#[derive(Debug, Clone)]
struct VariableMetadata {
    type_index: TypeIndex,
    variable_type: VariableType,
    // If the variable is a function we check for parameters.
    parameters: Option<Types>,
}

impl VariableMetadata {
    pub fn new(type_index: TypeIndex, variable_type: VariableType) -> Self {
        Self {
            type_index,
            variable_type,
            parameters: None,
        }
    }

    pub fn new_function(type_index: TypeIndex, parameters: Types) -> Self {
        Self {
            type_index,
            variable_type: VariableType::Function,
            parameters: Some(parameters),
        }
    }
}

struct Scope<'a> {
    prev: Option<Rc<RefCell<Scope<'a>>>>,
    nlocals: i64,
    names: HashMap<&'a str, VariableMetadata>,
    save: i64,
    loop_start: i64,
    loop_end: i64,
}

impl<'a> Scope<'a> {
    fn new(prev: Option<Rc<RefCell<Scope<'a>>>>) -> Self {
        let mut scope = Self {
            prev,
            nlocals: 0,
            names: HashMap::new(),
            save: 0,
            loop_end: -1,
            loop_start: -1,
        };

        if let Some(ref p) = scope.prev {
            scope.loop_start = p.borrow().loop_start;
            scope.loop_start = p.borrow().loop_end;
        }

        scope
    }

    fn get_var(&self, name: &str) -> Result<VariableMetadata> {
        match self.names.get(name) {
            Some(val) => Ok(val.to_owned()),
            None => {
                if let Some(prev_scope) = &self.prev {
                    prev_scope.borrow().get_var(name)
                } else {
                    Err(anyhow!(IrErrors::VariableNotFound))
                }
            }
        }
    }
}

pub struct Func<'a> {
    scope: Rc<RefCell<Scope<'a>>>,
    nvar: i64,
    stack: i64,
    prev: Option<Rc<RefCell<Func<'a>>>>,
    level: i64,
    return_type: Option<Types>,
    funcs: Vec<usize>, // the id's in the IrContext
    index: usize,
    pub labels: Vec<Option<i64>>,
    pub instructions: Vec<Instruction>,
}

impl<'a> Func<'a> {
    pub fn new(prev_fn: Option<Rc<RefCell<Func<'a>>>>) -> Self {
        let mut func = Self {
            scope: Rc::new(RefCell::new(Scope::new(None))),
            instructions: Vec::new(),
            stack: 0,
            nvar: 0,
            level: 0,
            return_type: None,
            prev: prev_fn,
            funcs: Vec::new(),
            labels: Vec::new(),
            index: 0,
        };

        if let Some(ref p) = func.prev {
            let borrow = p.borrow();
            func.level = borrow.level + 1;
            func.funcs = borrow.funcs.clone()
        }

        func
    }

    fn tmp(&mut self) -> i64 {
        let dst = self.stack;
        self.stack += 1;
        dst
    }

    // returns the address where the variable resides.
    fn add_var(&mut self, name: &'a str, ty: Types, variable_type: VariableType) -> Result<i64> {
        if self.scope.borrow().names.contains_key(name) {
            Err(anyhow!(IrErrors::DuplicateName))
        } else {
            let mut scope_mut = self.scope.borrow_mut();
            scope_mut
                .names
                .insert(name, VariableMetadata::new((ty, self.nvar), variable_type));
            scope_mut.nlocals += 1;
            assert!((self.stack == self.nvar), "stack is not equal to nvar");
            let dst = self.stack;
            self.stack += 1;
            self.nvar += 1;
            Ok(dst)
        }
    }

    fn new_label(&mut self) -> i64 {
        let label = self.labels.len() as i64;
        self.labels.push(None);
        label
    }

    fn set_label(&mut self, label: i64) {
        assert!(
            (label < self.labels.len() as i64),
            "label index too big for labels array"
        );
        self.labels[label as usize] = Some(self.instructions.len() as i64);
    }

    fn enter_scope(&mut self) {
        self.scope = Rc::new(RefCell::new(Scope::new(Some(self.scope.clone()))));
        self.scope.borrow_mut().save = self.stack;
    }

    fn leave_scope(&mut self) {
        self.stack = self.scope.borrow().save;
        self.nvar -= self.scope.borrow().nlocals;
        let prev_scope = { self.scope.borrow().prev.clone() };
        if let Some(prev_scope) = prev_scope {
            self.scope = prev_scope;
        }
    }

    fn mov(&mut self, a: i64, b: i64) -> i64 {
        if a != b {
            self.instructions.push(Instruction::Mov(a, b))
        }
        b
    }

    fn get_var(&self, name: &'a str) -> Result<VariableMetadata> {
        self.scope.borrow().get_var(name)
    }
}

fn has_common_type(vec1: &Vec<Type>, vec2: &Vec<Type>) -> bool {
    let set1: HashSet<_> = vec1.iter().cloned().collect();
    vec2.iter().any(|item| set1.contains(item))
}

pub struct IrContext<'a> {
    pub funcs: Vec<Rc<RefCell<Func<'a>>>>, // functions in a given context.
    pub curr: Rc<RefCell<Func<'a>>>,

    fn_name_to_idx: HashMap<&'a str, usize>,
}

impl<'a> IrContext<'a> {
    pub fn new(starting_func: Rc<RefCell<Func<'a>>>) -> Self {
        let mut ir_context = Self {
            curr: starting_func.clone(),
            funcs: Vec::new(),
            fn_name_to_idx: HashMap::new(),
        };

        ir_context.funcs.push(starting_func);
        ir_context
    }

    pub fn new_func(&mut self, fn_data: Rc<RefCell<Func<'a>>>) {
        self.funcs.push(fn_data);
    }

    // emit to current function
    pub fn emit(&self, instruction: Instruction) {
        self.curr.borrow_mut().instructions.push(instruction);
    }

    pub fn comp_expr(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
        let stack = {
            let curr_fn = self.curr.borrow();
            if allow_var {
                assert!(
                    (curr_fn.stack == curr_fn.nvar),
                    "stack != nvar when allow_var"
                );
            }
            curr_fn.stack
        };

        let save = stack;
        let type_index = self.comp_expr_tmp(sexpr, allow_var)?;
        let var = type_index.1;
        let (stack, nvar) = {
            let f = self.curr.borrow();
            (f.stack, f.nvar)
        };
        assert!((var < stack), "returned addr outside of stack.");

        // Discard temporaries from the compilation above. Either the stack is local variables only
        // or we revert it back to it's original state.
        let nvar = nvar;
        {
            self.curr.borrow_mut().stack = if allow_var { nvar } else { save };
        }

        // The result is either a temporary stored at the top of the stack or a local variable.
        assert!((var <= stack), "returned addr outside of stack.");
        Ok(type_index)
    }

    fn comp_list(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;
        let arg = list[0].as_str()?;

        if list.len() == 3 {
            if let Ok(binary_op) = arg.parse::<Binop>() {
                return self.comp_binop(sexpr, binary_op);
            }
        }

        if list.len() == 2 {
            if let Ok(op) = arg.parse::<Unop>() {
                return self.comp_unop(sexpr, op);
            }
        }

        match arg {
            "do" | "then" | "else" => self.comp_scope(sexpr),
            "var" | "const" if list.len() == 3 => {
                if !allow_var {
                    Err(anyhow!(IrErrors::ForbiddenVariableDeclaration))
                } else {
                    self.comp_newvar(sexpr, arg == "const")
                }
            }
            "set" if list.len() == 3 => self.comp_setvar(sexpr),
            "if" if list.len() == 3 || list.len() == 4 => self.comp_cond(sexpr),
            "break" if list.len() == 1 => {
                if self.curr.borrow().scope.borrow().loop_end < 0 {
                    Err(anyhow!(IrErrors::BreakOutsideLoop))
                } else {
                    self.emit(Instruction::Jmp(self.curr.borrow().scope.borrow().loop_end));
                    Ok((vec![Type::Void], -1))
                }
            }
            "continue" if list.len() == 1 => {
                if self.curr.borrow().scope.borrow().loop_start < 0 {
                    Err(anyhow!(IrErrors::ContinueOutsideLoop))
                } else {
                    self.emit(Instruction::Jmp(
                        self.curr.borrow().scope.borrow().loop_start,
                    ));
                    Ok((vec![Type::Void], -1))
                }
            }
            "loop" if list.len() == 3 => self.comp_loop(sexpr),
            "call" if list.len() >= 2 => self.comp_call(sexpr),
            _ => {
                println!("{:?}", sexpr);
                Err(anyhow!(IrErrors::UnknownExpression))
            }
        }
    }

    fn comp_loop(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;

        let (loop_start, loop_end) = {
            let mut curr_fn_mut = self.curr.borrow_mut();
            let loop_start_label = curr_fn_mut.new_label();
            let loop_end_label = curr_fn_mut.new_label();

            let mut scope_borrow = curr_fn_mut.scope.borrow_mut();
            scope_borrow.loop_start = loop_start_label;
            scope_borrow.loop_end = loop_end_label;

            (loop_start_label, loop_end_label)
        };

        {
            let mut fn_borrow = self.curr.borrow_mut();
            fn_borrow.enter_scope();
            fn_borrow.set_label(loop_start);
        }

        let (_, var) = self.comp_expr(&list[1], true)?;
        if var < 0 {
            return Err(anyhow!(IrErrors::BadConditionType));
        }
        {
            self.emit(Instruction::Jmpf(var, loop_end));
        }

        {
            let fn_borrow = self.curr.borrow();
            let scope_borrow = fn_borrow.scope.borrow_mut();
            println!("{} {}", scope_borrow.loop_start, scope_borrow.loop_end);
        }

        self.comp_expr(&list[2], false)?;
        {
            self.emit(Instruction::Jmp(loop_start));
        }

        let mut fn_mut = self.curr.borrow_mut();
        fn_mut.set_label(loop_end);
        fn_mut.leave_scope();

        Ok((vec![Type::Void], -1))
    }

    fn comp_scope(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;
        {
            self.curr.borrow_mut().enter_scope();
        }
        // clone this such that we can return to the parent function context later.
        let paren_func = self.curr.clone();

        let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
        for (idx, child) in list[1..].iter().enumerate() {
            let last_index = groups.len() - 1;
            groups[last_index].push(idx + 1);

            if let Ok(child_list) = child.as_list() {
                if child_list[0].as_str()? == "var" {
                    groups.push(Vec::new());
                }
            }
        }

        let (mut tp, mut var) = (vec![Type::Void], -1 as i64);
        // This is done to efficiently access lists. Without having to continually change convert types.
        for group in groups {
            // TODO: optimize this and make this cleaner

            // preprocess all functions
            let scanned_functions: Vec<(Rc<RefCell<Func<'a>>>, usize)> = group
                .iter()
                .filter_map(|e| match list[*e].is_func_def() {
                    Ok(true) => Some((self.scan_func(&list[*e]).unwrap(), *e)),
                    _ => None,
                })
                .collect();

            for child in group {
                if let Some(func) = scanned_functions.iter().find(|e| e.1 == child) {
                    // Switch context and compile the function body.
                    self.curr = func.0.clone();
                    (tp, var) = self.comp_func(&list[child])?;
                    // Go back to outer function context.
                    self.curr = paren_func.clone();
                } else {
                    // Normal expression
                    (tp, var) = self.comp_expr(&list[child], true)?;
                }
            }
        }

        let mut fn_borrow = self.curr.borrow_mut();
        fn_borrow.leave_scope();

        // The return is either a local variable or a new temporary
        if var >= fn_borrow.stack {
            let t = fn_borrow.tmp();
            var = fn_borrow.mov(var, t);
        }

        Ok((tp, var))
    }

    fn comp_newvar(&mut self, sexpr: &SExp<'a>, is_const: bool) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;
        let ti = self.comp_expr(&list[2], false)?;
        if ti.1 < 0 {
            Err(anyhow!(IrErrors::BadVariableInitType))
        } else {
            let mut fn_mut = self.curr.borrow_mut();
            let variable_type = if is_const {
                VariableType::Const
            } else {
                VariableType::Var
            };
            let dst = fn_mut.add_var(list[1].as_str()?, ti.0.clone(), variable_type)?;
            Ok((ti.0, fn_mut.mov(ti.1, dst)))
        }
    }

    fn comp_setvar(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;

        let var_metadata = {
            let fn_mut = self.curr.borrow_mut();
            fn_mut.get_var(list[1].as_str()?)?
        };

        // cannot assign to functions nor const variables
        if var_metadata.variable_type == VariableType::Const
            || var_metadata.variable_type == VariableType::Function
        {
            return Err(anyhow!(IrErrors::AssignToConst));
        }

        let (dst_type, dst) = var_metadata.type_index;

        let (tp, var) = self.comp_expr(&list[2], false)?;
        if dst_type != tp {
            Err(anyhow!(IrErrors::BadVariableSetType))
        } else {
            let mut fn_mut = self.curr.borrow_mut();
            Ok((dst_type, fn_mut.mov(var, dst)))
        }
    }

    fn comp_expr_tmp(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
        match sexpr {
            SExp::List(_) => self.comp_list(sexpr, allow_var),
            SExp::F64(num) => {
                let dst = { self.curr.borrow_mut().tmp() };
                self.emit(Instruction::Constant(*num as i64, dst));
                Ok((vec![Type::Int, Type::Byte, Type::BytePtr], dst))
            }
            SExp::Str(s) => Ok(self.curr.borrow().get_var(s)?.type_index),
        }
    }

    fn comp_cond(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;
        let (label_true, label_false) = {
            let mut fn_mut = self.curr.borrow_mut();
            let label_true = fn_mut.new_label();
            let label_false = fn_mut.new_label();
            fn_mut.enter_scope();

            (label_true, label_false)
        };
        let (tp, var) = self.comp_expr(&list[1], true)?;
        if tp == vec![Type::Void] {
            return Err(anyhow!(IrErrors::ExpectedBooleanCondition));
        }
        {
            self.emit(Instruction::Jmpf(var, label_false));
        }

        let (t1, a1) = self.comp_expr(&list[2], false)?;
        if a1 >= 0 {
            let mut fn_mut = self.curr.borrow_mut();
            let st = fn_mut.stack;
            fn_mut.mov(a1, st);
        }

        let has_else = list.len() == 4;
        let (mut t2, mut a2) = (vec![Type::Void], -1);
        if has_else {
            self.emit(Instruction::Jmp(label_true));
        }
        {
            self.curr.borrow_mut().set_label(label_false);
        }
        if has_else {
            (t2, a2) = self.comp_expr(&list[3], false)?;
            if a2 >= 0 {
                let mut fn_mut = self.curr.borrow_mut();
                let st = fn_mut.stack;
                fn_mut.mov(a2, st);
            }
        }

        let mut fn_mut = self.curr.borrow_mut();
        fn_mut.set_label(label_true);
        fn_mut.leave_scope();

        if a1 < 0 || a2 < 0 || !has_common_type(&t1, &t2) {
            Ok((vec![Type::Void], -1))
        } else {
            Ok((t1, fn_mut.tmp()))
        }
    }

    fn comp_binop(&mut self, sexpr: &SExp<'a>, operation: Binop) -> Result<TypeIndex> {
        // As with most compilation functions this is already ensured to work.
        let list = sexpr.as_list()?;
        let save = { self.curr.borrow().stack };

        // If they're both numbers we can optimize it the same as loding a constant
        if let (Ok(lhs_number), Ok(rhs_number)) = (list[1].is_number(), list[2].is_number()) {
            let dst = { self.curr.borrow_mut().tmp() };
            self.emit(Instruction::Constant(
                calc_binop(operation, lhs_number as i64, rhs_number as i64)?,
                dst,
            ));
            return Ok((vec![Type::Int, Type::Byte, Type::BytePtr], dst));
        }

        // First element is the argument so that 1 and 2 are lhs and rhs.
        let lhs = self.comp_expr_tmp(&list[1], false)?;
        let rhs = self.comp_expr_tmp(&list[2], false)?;

        {
            self.curr.borrow_mut().stack = save;
        }
        if !(has_common_type(&lhs.0, &rhs.0) && (lhs.0[0] == Type::Int || lhs.0[0] == Type::Byte)) {
            Err(anyhow!(IrErrors::BadBinopTypes))
        } else {
            let dst = { self.curr.borrow_mut().tmp() };
            let byte_exp = lhs.0 == rhs.0 && lhs.0 == vec![Type::Byte];
            self.emit(Instruction::Binop(operation, lhs.1, rhs.1, dst, byte_exp));
            Ok((lhs.0, dst))
        }
    }

    fn comp_unop(&mut self, sexpr: &SExp<'a>, operation: Unop) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;

        let (mut t1, a1) = self.comp_expr(&list[1], false)?;
        match operation {
            Unop::Minus => {
                if t1[0] != Type::Int || t1[0] != Type::Byte {
                    return Err(anyhow!(IrErrors::BadUnopTypes));
                }
            }
            Unop::Not => {
                if t1[0] != Type::Int || t1[0] != Type::Byte || t1[0] != Type::BytePtr {
                    return Err(anyhow!(IrErrors::BadUnopTypes));
                }
                t1 = vec![Type::Int];
            }
        }

        let mut fn_borrow = self.curr.borrow_mut();
        let dst = fn_borrow.tmp();
        fn_borrow
            .instructions
            .push(Instruction::Unop(operation, a1, dst));
        Ok((t1, dst))
    }

    fn comp_call(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let list = sexpr.as_list()?;
        let fn_name = list[1].as_str()?;

        let mut arg_types: Vec<Types> = Vec::new();
        for child in &list[2..] {
            let (tp, var) = self.comp_expr(child, false)?;
            arg_types.push(tp);
            let mut fn_borrow = self.curr.borrow_mut();
            let tmp = fn_borrow.tmp();
            fn_borrow.mov(var, tmp);
        }
        let mut fn_borrow = self.curr.borrow_mut();
        fn_borrow.stack -= arg_types.len() as i64;
        // let fn_metadata = fn_borrow.scope.borrow().get_var(fn_name)?;
        // let parameter_types = fn_metadata.parameters.unwrap();
        //
        // // check that types match
        // if !arg_types
        //     .iter()
        //     .enumerate()
        //     .all(|(index, ref arg_type)| arg_type.contains(&parameter_types[index]))
        // {
        //     return Err(anyhow!(IrErrors::InvalidFunctionArguments));
        // }

        let call_fn_idx = self
            .fn_name_to_idx
            .get(fn_name)
            .ok_or(anyhow!(IrErrors::FunctionNotFound))?;

        if *call_fn_idx == fn_borrow.index {
            let (fn_stack, fn_level) = (fn_borrow.stack, fn_borrow.level);
            fn_borrow.instructions.push(Instruction::Call(
                *call_fn_idx as i64,
                fn_stack,
                fn_level,
                fn_level,
            ));
            let mut dst: i64 = -1;
            if fn_borrow.return_type.clone().unwrap() != vec![Type::Void] {
                dst = fn_borrow.tmp();
            }
            Ok((fn_borrow.return_type.clone().unwrap(), dst))
        } else {
            let call_fn_borrow = self.funcs[*call_fn_idx].borrow_mut();
            let call_fn_level = call_fn_borrow.level;
            let (fn_stack, fn_level) = (fn_borrow.stack, fn_borrow.level);

            fn_borrow.instructions.push(Instruction::Call(
                *call_fn_idx as i64,
                fn_stack,
                fn_level,
                call_fn_level,
            ));

            let mut dst: i64 = -1;
            if call_fn_borrow.return_type.clone().unwrap() != vec![Type::Void] {
                dst = fn_borrow.tmp();
            }

            Ok((call_fn_borrow.return_type.clone().unwrap(), dst))
        }
    }

    pub fn scan_func(&mut self, sexpr: &SExp<'a>) -> Result<Rc<RefCell<Func<'a>>>> {
        let list = sexpr.as_list()?;
        let fn_info = list[1].as_list()?;
        let fn_name = fn_info[0].as_str()?;
        let ty = fn_info[1].as_str()?.parse::<Type>()?;

        // Scan argument types
        let arguments = list[2].as_list()?;
        let parameter_types: Result<Vec<Type>, anyhow::Error> = arguments
            .iter()
            .map(|arg| {
                arg.as_list()
                    .and_then(|l| l[1].as_str())
                    .and_then(|str_type| str_type.parse::<Type>())
            })
            .collect();

        let parameter_types = match parameter_types {
            Ok(types) => types,
            Err(e) => return Err(e),
        };

        {
            let func = self.curr.borrow();
            func.scope.borrow_mut().names.insert(
                fn_name,
                VariableMetadata::new_function(
                    (vec![ty.clone()], func.funcs.len() as i64),
                    parameter_types,
                ),
            );
        }

        let mut scanned_func = Func::new(Some(self.curr.clone()));
        scanned_func.return_type = Some(vec![ty]);
        scanned_func.index = self.funcs.len();
        let scanned_func = Rc::new(RefCell::new(scanned_func));
        self.new_func(scanned_func.clone());
        self.fn_name_to_idx.insert(fn_name, self.funcs.len() - 1);

        Ok(scanned_func)
    }

    // comp_func handles creating the ir for a given function. It sets the current function field in the ir context.
    pub fn comp_func(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
        let fn_expr = sexpr.as_list()?;
        let arguments_list = fn_expr[2].as_list()?;
        {
            // Function arguments are treated the same as local variables.
            let mut fn_borrow = self.curr.borrow_mut();
            for exp in arguments_list.iter().map(|e| e.as_list().unwrap()) {
                let ty = exp[1].as_str()?.parse::<Type>()?;
                fn_borrow.add_var(exp[0].as_str()?, vec![ty], VariableType::Function)?;
            }

            assert!(
                fn_borrow.stack as usize == arguments_list.len(),
                "function stack is not equal to parameter size"
            )
        }

        // Compile the function body and ensure that the types match.
        let (body_type, mut addr) = self.comp_expr(&fn_expr[3], false)?;
        let mut fn_borrow = self.curr.borrow_mut();

        if let Some(ret_type) = fn_borrow.return_type.clone() {
            if ret_type != vec![Type::Void] && !has_common_type(&body_type, &ret_type) {
                return Err(anyhow!(IrErrors::BadBodyType));
            }

            if ret_type == vec![Type::Void] {
                addr = -1;
            }

            fn_borrow.instructions.push(Instruction::Ret(addr));
            Ok((vec![Type::Void], -1))
        } else {
            Err(anyhow!(IrErrors::MissingReturnType))
        }
    }

    // dump_instructions dumps a display of the ir instructions into a given writer.
    pub fn dump_instructions<W: Write>(&self, mut writer: W) -> Result<()> {
        // loop over every function skipping the starting function.
        for (idx, func) in self.funcs[1..].iter().enumerate() {
            let fn_borrow = func.borrow();
            writeln!(writer, "func{}:", idx)?;
            let mut positions_to_labels: HashMap<usize, Vec<usize>> = HashMap::new();
            for (label, pos) in fn_borrow.labels.iter().enumerate() {
                positions_to_labels
                    .entry(pos.unwrap() as usize)
                    .or_insert_with(Vec::new)
                    .push(label);
            }

            // Check if a label exists as the given position.
            for (pos, instr) in fn_borrow.instructions.iter().enumerate() {
                match positions_to_labels.get(&pos) {
                    Some(labels) => {
                        for l in labels {
                            writeln!(writer, "L{}:", *l)?;
                        }
                    }
                    None => {}
                }

                writeln!(writer, "    {}", instr)?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}
