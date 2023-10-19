use std::collections::HashMap;

#[derive(Copy, Clone)]
enum Type {
  Int,
  Byte,
  IntPtr,
  BytePtr,
  Void,
}

struct Scope<'a> {
  prev: Option<Box<Scope<'a>>>,
  nlocals: usize,
  names: HashMap<&'a str, (Type, usize)>,
}

struct Func<'a> {
  scope: Scope<'a>,
  nvar: usize,
  stack: usize,
}

impl<'a> Scope<'a> {
  fn get_var(&self, name: &str) -> Option<(Type, usize)> {
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
