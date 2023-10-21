use crate::sexpr::*;
use anyhow::anyhow;
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
enum IrErrors {
  DuplicateName,
  UnknownExpression,
  BadBinopTypes,
  ForbiddenVariableDeclaration,
  BadVariableInitType,
  BadVariableSetType,
  VariableNotFound,
}

impl std::fmt::Display for IrErrors {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::DuplicateName => write!(f, "Duplicate variable name"),
      Self::UnknownExpression => write!(f, "unknown expression"),
      Self::BadBinopTypes => write!(f, "Unsuitable binop types"),
      Self::ForbiddenVariableDeclaration => write!(f, "Forbidden variable declaration"),
      Self::BadVariableInitType => write!(f, "Bad variable init type"),
      Self::VariableNotFound => write!(f, "Variable not found"),
      Self::BadVariableSetType => write!(f, "Bad variable set type"),
    }
  }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
  Int,
  Byte,
  BytePtr,
  Void,
}

#[derive(PartialEq, Debug)]
pub enum Binop {
  Plus,
  Minus,
  Multiply,
  Divide,
}

impl std::str::FromStr for Binop {
  type Err = anyhow::Error;

  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s {
      "+" => Ok(Self::Plus),
      "-" => Ok(Self::Minus),
      "/" => Ok(Self::Divide),
      "*" => Ok(Self::Multiply),
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
    }
  }
}

#[derive(PartialEq, Debug)]
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
  Label(i64),
  Binop(Binop, i64, i64, i64, bool),
  Mov(i64, i64),
}

impl std::fmt::Display for Instruction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Constant(a, b) => write!(f, "const {} {}", a, b),
      Self::Label(l) => write!(f, "label {}", l),
      Self::Binop(ty, lhs, rhs, dst, is_byte) => {
        if *is_byte {
          write!(f, "binop8 {} {} {} {}", ty, lhs, rhs, dst)
        } else {
          write!(f, "binop {} {} {} {}", ty, lhs, rhs, dst)
        }
      }
      Self::Mov(a, b) => write!(f, "mov {} {}", a, b),
    }
  }
}

type Types = Vec<Type>;
type TypeIndex = (Types, i64);

struct Scope<'a> {
  prev: Option<Rc<RefCell<Scope<'a>>>>,
  nlocals: i64,
  names: HashMap<&'a str, TypeIndex>,
  save: i64,
}

pub struct Func<'a> {
  scope: Rc<RefCell<Scope<'a>>>,
  nvar: i64,
  stack: i64,
  labels: Vec<Option<i64>>,
  pub instructions: Vec<Instruction>,
}

impl<'a> Scope<'a> {
  fn new(prev: Option<Rc<RefCell<Scope<'a>>>>) -> Self {
    Self {
      prev,
      nlocals: 0,
      names: HashMap::new(),
      save: 0,
    }
  }

  fn get_var(&self, name: &str) -> Result<TypeIndex> {
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

#[macro_export]
macro_rules! check_len {
  ($list:expr, $n:expr) => {
    if $list.len() != $n {
      Err(anyhow!(IrErrors::UnknownExpression))
    } else {
      Ok(())
    }
  };
}

impl<'a> Func<'a> {
  pub fn new() -> Self {
    Self {
      scope: Rc::new(RefCell::new(Scope::new(None))),
      instructions: Vec::new(),
      stack: 0,
      nvar: 0,
      labels: Vec::new(),
    }
  }

  fn tmp(&mut self) -> i64 {
    let dst = self.stack;
    self.stack += 1;
    dst
  }

  // returns the address where the variable resides.
  fn add_var(&mut self, name: &'a str, ty: Types) -> Result<i64> {
    if self.scope.borrow().names.contains_key(name) {
      Err(anyhow!(IrErrors::DuplicateName))
    } else {
      self.scope.borrow_mut().names.insert(name, (ty, self.nvar));
      self.scope.borrow_mut().nlocals += 1;
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
    assert!((label < self.labels.len() as i64), "label index too big for labels array");
    self.labels[label as usize] = Some(self.instructions.len() as i64);
  }

  pub fn comp_expr(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
    if allow_var {
      assert!((self.stack == self.nvar), "stack != nvar when allow_var");
    }

    let save = self.stack;
    let type_index = self.comp_expr_tmp(sexpr, allow_var)?;
    let var = type_index.1;
    assert!((var < self.stack), "returned addr outside of stack.");

    // Discard temporaries from the compilation above. Either the stack is local variables only
    // or we revert it back to it's original state.
    self.stack = if allow_var { self.nvar } else { save };

    // The result is either a temporary stored at the top of the stack or a local variable.
    assert!((var <= self.stack), "returned addr outside of stack.");
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
      if let Ok(_unary_op) = arg.parse::<Unop>() {
        // TODO: unop
      }
    }

    match arg {
      "do" | "then" | "else" => self.comp_scope(sexpr),
      "var" => {
        check_len!(list, 3)?;
        if !allow_var {
          Err(anyhow!(IrErrors::ForbiddenVariableDeclaration))
        } else {
          self.comp_newvar(sexpr)
        }
      }
      "set" => {
        check_len!(list, 3)?;
        self.comp_setvar(sexpr)
      }
      _ => Err(anyhow!(IrErrors::UnknownExpression)),
    }
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

  fn comp_scope(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
    let list = sexpr.as_list()?;

    self.enter_scope();
    let (mut tp, mut var) = (vec![Type::Void], -1 as i64);
    for child in &list[1..] {
      (tp, var) = self.comp_expr(child, true)?;
    }
    self.leave_scope();

    // The return is either a local variable or a new temporary
    if var >= self.stack {
      let t = self.tmp();
      var = self.mov(var, t);
    }

    Ok((tp, var))
  }

  fn comp_newvar(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
    let list = sexpr.as_list()?;

    let ti = self.comp_expr(&list[2], false)?;
    if ti.1 < 0 {
      Err(anyhow!(IrErrors::BadVariableInitType))
    } else {
      let dst = self.add_var(list[1].as_str()?, ti.0.clone())?;
      Ok((ti.0, self.mov(ti.1, dst)))
    }
  }

  fn comp_setvar(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
    let list = sexpr.as_list()?;

    let (dst_type, dst) = self.get_var(list[1].as_str()?)?;
    let (tp, var) = self.comp_expr(&list[2], false)?;
    if dst_type != tp {
      Err(anyhow!(IrErrors::BadVariableSetType))
    } else {
      Ok((dst_type, self.mov(var, dst)))
    }
  }

  fn comp_expr_tmp(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
    match sexpr {
      SExp::List(_) => self.comp_list(sexpr, allow_var),
      SExp::F64(num) => {
        let dst = self.tmp();
        self
          .instructions
          .push(Instruction::Constant(*num as i64, dst));
        Ok((vec![Type::Int, Type::Byte, Type::BytePtr], dst))
      }
      SExp::Str(_) => todo!(),
    }
  }

  fn comp_binop(&mut self, sexpr: &SExp<'a>, operation: Binop) -> Result<TypeIndex> {
    // As with most compilation functions this is already ensured to work.
    let list = sexpr.as_list()?;
    let save = self.stack;
    // TODO: convert binop from string and handle boolean comparisons.

    // First element is the argument so that 1 and 2 are lhs and rhs.
    let lhs = self.comp_expr_tmp(&list[1], false)?;
    let rhs = self.comp_expr_tmp(&list[2], false)?;

    self.stack = save;
    if !(lhs.0 == rhs.0 && (lhs.0[0] == Type::Int || lhs.0[0] == Type::Byte)) {
      Err(anyhow!(IrErrors::BadBinopTypes))
    } else {
      let dst = self.tmp();
      let byte_exp = lhs.0 == rhs.0 && lhs.0 == vec![Type::Byte];
      self
        .instructions
        .push(Instruction::Binop(operation, lhs.1, rhs.1, dst, byte_exp));
      Ok((lhs.0, dst))
    }
  }

  fn mov(&mut self, a: i64, b: i64) -> i64 {
    if a != b {
      self.instructions.push(Instruction::Mov(a, b))
    }
    b
  }

  fn get_var(&self, name: &'a str) -> Result<TypeIndex> {
    self.scope.borrow().get_var(name)
  }
}
