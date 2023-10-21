use crate::sexpr::*;
use anyhow::anyhow;
use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug)]
enum IrErrors {
  DuplicateName,
  UnknownExpression,
  BadBinopTypes,
}

impl std::fmt::Display for IrErrors {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::DuplicateName => write!(f, "Duplicate variable name"),
      Self::UnknownExpression => write!(f, "unknown expression"),
      Self::BadBinopTypes => write!(f, "Unsuitable binop types"),
    }
  }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
  Int,
  Byte,
  IntPtr,
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

#[derive(PartialEq, Debug)]
pub enum Unop {
  Minus,
  Not,
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
      Self::Binop(_, lhs, rhs, dst, is_byte) => {
        if *is_byte {
          write!(f, "binop8 TODO {} {} {}", lhs, rhs, dst)
        } else {
          write!(f, "binop TODO {} {} {}", lhs, rhs, dst)
        }
      }
      Self::Mov(a, b) => write!(f, "mov {} {}", a, b),
    }
  }
}

type Types = Vec<Type>;
type TypeIndex = (Types, i64);

struct Scope<'a> {
  prev: Option<Box<Scope<'a>>>,
  nlocals: i64,
  names: HashMap<&'a str, TypeIndex>,
}

pub struct Func<'a> {
  scope: Scope<'a>,
  nvar: i64,
  stack: i64,
  pub instructions: Vec<Instruction>,
}

impl<'a> Scope<'a> {
  fn new(prev: Option<Box<Scope<'a>>>) -> Self {
    Self {
      prev,
      nlocals: 0,
      names: HashMap::new(),
    }
  }

  fn get_var(&self, name: &str) -> Option<TypeIndex> {
    match self.names.get(name) {
      Some(val) => Some(val.to_owned()),
      None => {
        if let Some(prev_scope) = &self.prev {
          prev_scope.get_var(name)
        } else {
          None
        }
      }
    }
  }
}

impl<'a> Func<'a> {
  pub fn new() -> Self {
    Self {
      scope: Scope::new(None),
      instructions: Vec::new(),
      stack: 0,
      nvar: 0,
    }
  }

  fn tmp(&mut self) -> i64 {
    let dst = self.stack;
    self.stack += 1;
    dst
  }

  // returns the address where the variable resides.
  fn add_var(&mut self, name: &'a str, ty: Types) -> Result<i64> {
    if self.scope.names.contains_key(name) {
      Err(anyhow!(IrErrors::DuplicateName))
    } else {
      self.scope.names.insert(name, (ty, self.nvar));
      self.scope.nlocals += 1;
      assert!((self.stack == self.nvar), "stack is not equal to nvar");
      let dst = self.stack;
      self.stack += 1;
      self.nvar += 1;
      Ok(dst)
    }
  }

  pub fn comp_expr(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
    if allow_var {
      assert!((self.stack == self.nvar), "stack != nvar when allow_var");
    }

    let save = self.stack;
    let type_index = self.comp_expr_tmp(sexpr, allow_var)?;
    let var = type_index.1;
    assert!(
      (var < self.stack),
      "returned addr outside of stack."
    );

    // Discard temporaries from the compilation above. Either the stack is local variables only
    // or we revert it back to it's original state.
    self.stack = if allow_var { self.nvar } else { save };

    // The result is either a temporary stored at the top of the stack or a local variable.
    assert!((var <= self.stack), "returned addr outside of stack.");
    Ok(type_index)
  }

  fn comp_expr_tmp(&mut self, sexpr: &SExp<'a>, allow_var: bool) -> Result<TypeIndex> {
    match sexpr {
      SExp::List(list) => {
        let arg = list[0].as_str()?;
        if arg == "+" && list.len() == 3 {
          self.comp_binop(sexpr)
        } else {
          Err(anyhow!(IrErrors::UnknownExpression))
        }
      }
      SExp::F64(num) => {
        let dst = self.tmp();
        self.instructions.push(Instruction::Constant(*num as i64, dst));
        Ok((vec![Type::Int, Type::Byte, Type::BytePtr], dst))
      }
      SExp::Str(_) => todo!()
    }
  }

  fn comp_binop(&mut self, sexpr: &SExp<'a>) -> Result<TypeIndex> {
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
      self.instructions.push(Instruction::Binop(Binop::Plus, lhs.1, rhs.1, dst, byte_exp));
      Ok((lhs.0, dst))
    }
  }
}
