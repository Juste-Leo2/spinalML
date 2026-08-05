import mill._
import mill.scalalib._

object spinalML extends ScalaModule {
  def scalaVersion = "2.12.18"

  def ivyDeps = Agg(
    ivy"com.github.spinalhdl::spinalhdl-core:1.14.2",
    ivy"com.github.spinalhdl::spinalhdl-lib:1.14.2"
  )

  def scalacPluginIvyDeps = Agg(ivy"com.github.spinalhdl::spinalhdl-idsl-plugin:1.14.2")

  object test extends ScalaTests with TestModule.ScalaTest {
    def ivyDeps = super.ivyDeps() ++ Agg(
      ivy"org.scalatest::scalatest:3.2.17"
    )
  }
}
